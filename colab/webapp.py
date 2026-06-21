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
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_sock import Sock
from werkzeug.utils import secure_filename

from . import generate

CONTENT_DIR = Path(os.environ.get("ZIPCAST_CONTENT_DIR", "/content"))
WORK_DIR = Path(os.environ.get("ZIPCAST_WORK_DIR", str(CONTENT_DIR / "zipcast_work")))
STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
sock = Sock(app)


class _SuppressActivePoll(logging.Filter):
    """the notebook's mirror loop polls /api/jobs/active once a second while
    idle, waiting for the next job -- silence just that access-log line so it
    doesn't bury everything else in repeated noise, while leaving real
    requests (and any errors) visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/jobs/active" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_SuppressActivePoll())

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None

# engines are loaded once and cached for the life of the process -- loading
# takes minutes, so reloading per job (or per voice-test click) would be a
# multi-minute tax every single time. keyed by (kind, model_name, device).
_engine_cache: dict[tuple[str, str, str], object] = {}
_engine_lock = threading.Lock()


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_or_load_engine(kind: str, model_name: str, device: str, on_progress=None):
    """kind is 'preset' (CustomVoice) or 'design' (VoiceDesign)."""
    key = (kind, model_name, device)
    with _engine_lock:
        engine = _engine_cache.get(key)
        if engine is None:
            if kind == "design":
                engine = generate.Qwen3VoiceDesignEngine(model_name=model_name, device=device, on_progress=on_progress)
            else:
                engine = generate.Qwen3TTSEngine(model_name=model_name, device=device, on_progress=on_progress)
            _engine_cache[key] = engine
        return engine


def preload(model_name: str = generate.DEFAULT_MODEL, device: str | None = None, on_progress=None) -> None:
    """load the named-speaker CustomVoice model into the cache now, so the
    first job started from the GUI doesn't pay the multi-minute load cost.
    call this before starting the server/tunnel.
    """
    device = device or _device()
    _get_or_load_engine("preset", model_name, device, on_progress=on_progress)


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


@app.route("/api/books/<path:filename>/chapters")
def api_book_chapters(filename: str):
    # continuation-numbered series (e.g. epub "volume 2" whose first chapter
    # title says "Chapter 270") make plain positional ranges ("1-10") error
    # prone -- this lets the GUI show real chapter titles to search/pick from
    safe_name = secure_filename(filename)
    epub_path = (CONTENT_DIR / safe_name).resolve()
    if epub_path.parent != CONTENT_DIR.resolve() or not epub_path.exists():
        return jsonify({"error": "unknown book"}), 404

    try:
        metadata = generate.peek_epub(epub_path)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500

    chapters = [{"index": c["index"], "title": c["title"]} for c in metadata["chapters"]]
    return jsonify({"filename": safe_name, "title": metadata["title"], "chapters": chapters})


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
    engine = _get_or_load_engine("design", model_name, device)
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
        # reuse the cached/preloaded engine instead of loading a fresh one for
        # every job -- loading takes minutes, this should be a one-time cost
        device = kwargs.get("device", "cpu")
        if kwargs.get("voice_mode") == "design":
            model_name = kwargs.get("design_model_name", generate.VOICE_DESIGN_MODEL)
            engine = _get_or_load_engine("design", model_name, device, on_progress=on_progress)
        else:
            model_name = kwargs.get("model_name", generate.DEFAULT_MODEL)
            engine = _get_or_load_engine("preset", model_name, device, on_progress=on_progress)

        generate.run_batch(
            content_dir=str(CONTENT_DIR), work_dir=str(WORK_DIR), engine=engine, on_progress=on_progress, **kwargs
        )
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


@sock.route("/ws/jobs/<job_id>")
def ws_job_events(ws, job_id: str):
    # websocket instead of SSE: SSE's chunked response was getting buffered
    # somewhere between the Cloudflare quick tunnel and the browser, so
    # progress only ever showed up in bursts (or not at all) instead of live.
    # replays the full event history from the start, so a page refresh that
    # reconnects here sees everything that already happened, not just what's
    # left.
    job = _jobs.get(job_id)
    if job is None:
        ws.send(json.dumps({"event": "error", "message": "unknown job_id"}))
        return

    idx = 0
    while True:
        with job["lock"]:
            job["cond"].wait_for(lambda: len(job["events"]) > idx or job["status"] != "running", timeout=15)
            new_events = job["events"][idx:]
            idx = len(job["events"])
            finished = job["status"] != "running"
        for event in new_events:
            ws.send(json.dumps(event))
        if finished and idx >= len(job["events"]):
            ws.send(json.dumps({"event": "stream_end", "status": job["status"]}))
            break


@app.route("/api/download/<path:filename>")
def api_download(filename: str):
    return send_from_directory(CONTENT_DIR, filename, as_attachment=True)


def start_server(port: int = 5000) -> None:
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


def start_server_in_background(port: int = 5000) -> threading.Thread:
    thread = threading.Thread(target=start_server, kwargs={"port": port}, daemon=True)
    thread.start()
    return thread
