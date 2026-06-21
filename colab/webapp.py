"""Small local web GUI for the zipcast Colab pipeline: lets you pick which
epub(s)/chapters to convert, tune chunk/batch size, preview a described voice,
and watch live progress -- instead of editing notebook cells by hand.

Run from a notebook cell (see notebooks/colab_zipcast_qwen3tts.ipynb):

    from colab.webapp import start_server_in_background
    start_server_in_background(port=5000)

then expose port 5000 with a tunnel (the notebook uses a Cloudflare quick
tunnel, which needs no account/auth) and open the printed URL.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from . import generate

CONTENT_DIR = Path(os.environ.get("ZIPCAST_CONTENT_DIR", "/content"))
WORK_DIR = Path(os.environ.get("ZIPCAST_WORK_DIR", str(CONTENT_DIR / "zipcast_work")))
STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None

_design_engine_cache: dict[str, "generate.Qwen3VoiceDesignEngine"] = {}
_design_engine_lock = threading.Lock()


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/books")
def api_books():
    books = []
    for epub_path in sorted(CONTENT_DIR.glob("*.epub")):
        try:
            metadata = generate.peek_epub(epub_path)
        except Exception as e:  # noqa: BLE001 - one bad epub shouldn't break the listing
            books.append({"filename": epub_path.name, "error": str(e)})
            continue
        books.append({
            "filename": epub_path.name,
            "title": metadata["title"],
            "author": metadata["author"],
            "chapters": len(metadata["chapters"]),
        })
    return jsonify({"books": books})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files in request"}), 400

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    rejected = []
    for f in files:
        filename = secure_filename(f.filename or "")
        if not filename:
            rejected.append({"filename": f.filename or "(unnamed)", "reason": "invalid filename"})
            continue
        if not filename.lower().endswith(".epub"):
            rejected.append({"filename": filename, "reason": "not a .epub file"})
            continue
        dest = CONTENT_DIR / filename
        f.save(dest)
        saved.append(filename)

    return jsonify({"saved": saved, "rejected": rejected})


@app.route("/api/defaults")
def api_defaults():
    return jsonify({
        "speakers": generate.BUILTIN_SPEAKERS,
        "default_speaker": generate.DEFAULT_SPEAKER,
        "model_name": generate.DEFAULT_MODEL,
        "design_model_name": generate.VOICE_DESIGN_MODEL,
        "chunk_chars": generate.DEFAULT_CHUNK_CHARS,
        "batch_size": generate.DEFAULT_BATCH_SIZE,
        "voice_test_text": generate.DEFAULT_VOICE_TEST_TEXT,
        "device": _device(),
    })


@app.route("/api/voice-test", methods=["POST"])
def api_voice_test():
    body = request.get_json(force=True) or {}
    description = (body.get("description") or "").strip()
    sample_text = (body.get("sample_text") or generate.DEFAULT_VOICE_TEST_TEXT).strip()
    model_name = body.get("model_name") or generate.VOICE_DESIGN_MODEL
    if not description:
        return jsonify({"error": "description is required"}), 400

    device = _device()
    with _design_engine_lock:
        engine = _design_engine_cache.get(model_name)
        if engine is None:
            engine = generate.Qwen3VoiceDesignEngine(model_name=model_name, device=device)
            _design_engine_cache[model_name] = engine
        try:
            audio, sr = engine.design_sample(sample_text, description)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


def _make_progress_recorder(job_id: str):
    job = _jobs[job_id]

    def on_progress(event: dict) -> None:
        with job["lock"]:
            job["events"].append(event)
            if event.get("event") == "book_done":
                job["outputs"].append(event["output_path"])
            job["cond"].notify_all()

    return on_progress


def _run_job(job_id: str, kwargs: dict) -> None:
    global _active_job_id
    job = _jobs[job_id]
    on_progress = _make_progress_recorder(job_id)
    try:
        # free GPU memory held by the voice-test engine before the real run
        with _design_engine_lock:
            for engine in _design_engine_cache.values():
                engine.unload()
            _design_engine_cache.clear()

        generate.run_batch(content_dir=str(CONTENT_DIR), work_dir=str(WORK_DIR), on_progress=on_progress, **kwargs)
        with job["lock"]:
            job["status"] = "done"
            job["cond"].notify_all()
    except Exception as e:  # noqa: BLE001
        with job["lock"]:
            job["status"] = "error"
            job["error"] = str(e)
            job["events"].append({"event": "error", "message": str(e)})
            job["cond"].notify_all()
    finally:
        with _jobs_lock:
            _active_job_id = None


@app.route("/api/jobs", methods=["POST"])
def api_start_job():
    global _active_job_id
    body = request.get_json(force=True) or {}

    with _jobs_lock:
        if _active_job_id is not None and _jobs[_active_job_id]["status"] == "running":
            return jsonify({"error": "a job is already running", "job_id": _active_job_id}), 409

        requested = body.get("epubs") or None
        epub_paths = None
        if requested:
            available = {p.name for p in CONTENT_DIR.glob("*.epub")}
            missing = [name for name in requested if name not in available]
            if missing:
                return jsonify({"error": f"epub(s) not found in {CONTENT_DIR}: {', '.join(missing)}"}), 400
            epub_paths = [str(CONTENT_DIR / name) for name in requested]

        kwargs = {
            "epub_paths": epub_paths,
            "chapters": body.get("chapters", "all"),
            "chunk_chars": int(body.get("chunk_chars", generate.DEFAULT_CHUNK_CHARS)),
            "batch_size": int(body.get("batch_size", generate.DEFAULT_BATCH_SIZE)),
            "device": _device(),
            "voice_mode": body.get("voice_mode", "preset"),
            "speaker": body.get("speaker", generate.DEFAULT_SPEAKER),
            "voice_description": body.get("voice_description", ""),
            "model_name": body.get("model_name", generate.DEFAULT_MODEL),
            "design_model_name": body.get("design_model_name", generate.VOICE_DESIGN_MODEL),
        }

        job_id = uuid.uuid4().hex[:12]
        lock = threading.Lock()
        _jobs[job_id] = {
            "status": "running",
            "events": [],
            "outputs": [],
            "error": None,
            "lock": lock,
            "cond": threading.Condition(lock),
            "started_at": time.time(),
        }
        _active_job_id = job_id

        thread = threading.Thread(target=_run_job, args=(job_id, kwargs), daemon=True)
        thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/jobs/active")
def api_active_job():
    with _jobs_lock:
        return jsonify({"job_id": _active_job_id})


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    with job["lock"]:
        return jsonify({
            "status": job["status"],
            "error": job["error"],
            "outputs": job["outputs"],
            "event_count": len(job["events"]),
        })


@app.route("/api/jobs/<job_id>/events")
def api_job_events(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404

    def stream():
        idx = 0
        while True:
            with job["lock"]:
                job["cond"].wait_for(lambda: len(job["events"]) > idx or job["status"] != "running", timeout=15)
                new_events = job["events"][idx:]
                idx = len(job["events"])
                finished = job["status"] != "running"
            if new_events:
                for event in new_events:
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                yield ": keepalive\n\n"
            if finished and idx >= len(job["events"]):
                yield f"data: {json.dumps({'event': 'stream_end', 'status': job['status']})}\n\n"
                break

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/download/<path:filename>")
def api_download(filename: str):
    return send_from_directory(CONTENT_DIR, filename, as_attachment=True)


def start_server(port: int = 5000) -> None:
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


def start_server_in_background(port: int = 5000) -> threading.Thread:
    thread = threading.Thread(target=start_server, kwargs={"port": port}, daemon=True)
    thread.start()
    return thread
