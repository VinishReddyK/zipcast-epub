"""Colab-side audiobook generation: find every .epub dropped into /content, parse
each directly (no manual zip/upload step), run Qwen3-TTS over its chapters, and
assemble the result into a single m4b with chapter markers and cover art.

Designed to be imported from the companion notebook or the web GUI
(see colab/webapp.py) after cloning this repo:

    from colab.generate import run_batch
    run_batch(on_progress=print)
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf  # type: ignore

SAMPLE_RATE = 24000
DEFAULT_CHUNK_CHARS = 500
DEFAULT_BATCH_SIZE = 10
PARAGRAPH_PAUSE_MS = 500
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_SPEAKER = "ryan"
BUILTIN_SPEAKERS = ["ryan", "vivian", "sunny", "aria", "bella", "nova", "echo", "finn", "atlas"]
DEFAULT_VOICE_TEST_TEXT = "Hello, this is a preview of how this voice sounds when reading your book."
METADATA_FILE = "metadata.json"
COVER_FILE = "cover.jpg"

ProgressCallback = Callable[[dict], None]


def _emit(on_progress: ProgressCallback | None, event: dict) -> None:
    if on_progress:
        on_progress(event)


def _log(on_progress: ProgressCallback | None, message: str) -> None:
    """print (for plain notebook/script use) and also emit a structured 'log'
    event so the web GUI -- which can't rely on seeing a background thread's
    stdout in the notebook cell output -- shows every message too.
    """
    print(message)
    _emit(on_progress, {"event": "log", "message": message})


def find_epubs(content_dir: str | Path = "/content") -> list[Path]:
    """locate every .epub file in content_dir, in name order."""
    content_dir = Path(content_dir)
    epub_paths = sorted(content_dir.glob("*.epub"))
    if not epub_paths:
        raise FileNotFoundError(
            f"no .epub files found in {content_dir}. drag one or more .epub files "
            "into the Colab Files panel on the left and re-run this cell."
        )
    return epub_paths


def peek_epub(epub_path: Path) -> dict:
    """parse epub_path's metadata + chapter list without writing anything to disk."""
    from zipcast.epub_parser import parse_epub

    book, _cover = parse_epub(epub_path)
    return book.to_metadata()


def extract_book(epub_path: Path, extract_dir: Path) -> dict:
    """parse epub_path directly and write its chapters/metadata/cover to extract_dir."""
    from zipcast.epub_parser import parse_epub

    extract_dir.mkdir(parents=True, exist_ok=True)
    book, cover_data = parse_epub(epub_path)
    if not book.chapters:
        raise ValueError(f"no chapters extracted from {epub_path} (epub may be malformed or DRM-protected)")

    metadata = book.to_metadata()
    (extract_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for chapter in book.chapters:
        (extract_dir / f"{chapter.filename_base}.txt").write_text(chapter.text, encoding="utf-8")
    if cover_data:
        (extract_dir / COVER_FILE).write_bytes(cover_data)
    return metadata


def parse_chapter_selection(spec: str, total_chapters: int) -> set[int]:
    """parse 'all' | '5' | '1-10' | '1,3,5-8' into a set of 1-based chapter indices."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return set(range(1, total_chapters + 1))

    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))

    indices = {i for i in indices if 1 <= i <= total_chapters}
    if not indices:
        raise ValueError(f"chapter selection '{spec}' matched no chapters (book has {total_chapters})")
    return indices


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """split chapter text into TTS-sized chunks on paragraph/sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) + 1 > max_chars:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            chunks.append(buf)
    return chunks


def concatenate_audio(chunks: list[np.ndarray], sample_rate: int, pause_ms: int) -> np.ndarray:
    """join synthesized chunks with a short silence between each."""
    if not chunks:
        return np.array([], dtype=np.float32)
    pause = np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        parts.append(np.asarray(chunk, dtype=np.float32))
        if i != len(chunks) - 1:
            parts.append(pause)
    return np.concatenate(parts)


def _select_dtype(torch, device: str):
    """bf16 only pays off on Ampere+ (compute capability >= 8.0, e.g. A100/L4);
    Turing cards like the free-tier Colab T4 (7.5) lack bf16 tensor cores, so
    fall back to fp16 there for real speed instead of just numerical compatibility.
    """
    if "cuda" not in device or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _quiet_transformers_logging() -> None:
    import transformers  # type: ignore

    # silences the "Setting `pad_token_id` to `eos_token_id`..." notice that
    # transformers' generate() logs on every call -- harmless, but with chunks
    # batched per-chapter this fires hundreds of times over a full book
    transformers.logging.set_verbosity_error()


class Qwen3TTSEngine:
    """built-in named-speaker engine (Qwen3-TTS CustomVoice model)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: ProgressCallback | None = None,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore

        _quiet_transformers_logging()
        self._torch = torch
        # cap how many chunks go into one generate_custom_voice() call: peak GPU
        # memory scales with batch size x max_new_tokens, so batching a whole
        # chapter's chunks at once OOMs on long chapters even on a 14-16GiB T4.
        # tune this down (e.g. to 2-4) if you still hit CUDA OOM.
        self.batch_size = max(1, batch_size)
        dtype = _select_dtype(torch, device)
        _log(on_progress, f"loading {model_name} on {device} ({dtype})... this can take a few minutes.")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        _log(on_progress, "model loaded.")

    def synthesize_batch(self, batch: list[str], speaker: str = DEFAULT_SPEAKER) -> tuple[list[np.ndarray], int]:
        with self._torch.inference_mode():
            wavs, sr = self.model.generate_custom_voice(
                text=batch, speaker=speaker, instruct="", max_new_tokens=2048
            )
        if isinstance(wavs, np.ndarray):
            wavs = [wavs]
        return list(wavs), sr

    def unload(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class Qwen3VoiceDesignEngine:
    """free-form voice-description engine (Qwen3-TTS VoiceDesign model).

    the underlying generate_voice_design() call only accepts one text at a
    time (no list-batching like CustomVoice), so synthesize_batch() here
    always processes one chunk per call regardless of the requested batch
    size -- this mode is meant for previewing/narrating with a described
    voice, not raw throughput.
    """

    def __init__(
        self,
        model_name: str = VOICE_DESIGN_MODEL,
        device: str = "cuda",
        on_progress: ProgressCallback | None = None,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore

        _quiet_transformers_logging()
        self._torch = torch
        self.batch_size = 1
        dtype = _select_dtype(torch, device)
        _log(on_progress, f"loading {model_name} on {device} ({dtype})... this can take a few minutes.")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        _log(on_progress, "model loaded.")

    def synthesize_batch(self, batch: list[str], instruct: str = "") -> tuple[list[np.ndarray], int]:
        results = []
        sr = SAMPLE_RATE
        with self._torch.inference_mode():
            for text in batch:
                wavs, sr = self.model.generate_voice_design(text=text, instruct=instruct)
                wav = wavs[0] if not isinstance(wavs, np.ndarray) else wavs
                results.append(wav)
        return results, sr

    def design_sample(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        wavs, sr = self.synthesize_batch([text], instruct=instruct)
        return wavs[0], sr

    def unload(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


@dataclass
class BookPlan:
    """a fully parsed + pre-chunked book, ready to synthesize. building this is
    CPU-only and cheap, so the whole batch is planned upfront -- that's what
    makes an accurate total-chunk-count (and therefore ETA) possible before
    any GPU work starts.
    """

    epub_path: Path
    metadata: dict
    extract_dir: Path
    wav_dir: Path
    output_path: Path
    selected_chapters: list[dict] = field(default_factory=list)
    chapter_chunks: dict[int, list[str]] = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        return sum(len(v) for v in self.chapter_chunks.values())


def plan_book(
    epub_path: Path,
    content_dir: Path,
    work_dir: Path,
    chapters_spec: str = "all",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> BookPlan:
    """parse one epub and pre-chunk its selected chapters, without doing any TTS."""
    book_dir = work_dir / epub_path.stem
    extract_dir = book_dir / "extract"
    wav_dir = book_dir / "wav"

    metadata = extract_book(epub_path, extract_dir)
    all_chapters = sorted(metadata["chapters"], key=lambda c: c["index"])
    selected_indices = parse_chapter_selection(chapters_spec, len(all_chapters))
    selected = [c for c in all_chapters if c["index"] in selected_indices]

    chapter_chunks: dict[int, list[str]] = {}
    for c in selected:
        text = (extract_dir / f"{c['filename_base']}.txt").read_text(encoding="utf-8")
        chapter_chunks[c["index"]] = chunk_text(text, max_chars=chunk_chars)

    safe_title = "".join(ch for ch in metadata["title"] if ch.isalnum() or ch in " -_").strip() or epub_path.stem
    output_path = content_dir / f"{safe_title}.m4b"

    return BookPlan(
        epub_path=epub_path,
        metadata=metadata,
        extract_dir=extract_dir,
        wav_dir=wav_dir,
        output_path=output_path,
        selected_chapters=selected,
        chapter_chunks=chapter_chunks,
    )


def _escape_ffmetadata(text: str) -> str:
    for c in ["=", ";", "#", "\\", "\n"]:
        text = text.replace(c, "\\" + c)
    return text


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "r") as f:
        return int(f.getnframes() / f.getframerate() * 1000)


def build_m4b(
    metadata: dict,
    wav_paths: list[Path],
    cover_path: Path | None,
    output_path: Path,
    bitrate: str = "192k",
    on_progress: ProgressCallback | None = None,
) -> Path:
    """concatenate per-chapter wavs into a single m4b with chapter markers + cover art."""
    ffmetadata = [";FFMETADATA1"]
    for key, value in {
        "title": metadata["title"],
        "artist": metadata["author"],
        "album": metadata["title"],
        "genre": "Audiobook",
    }.items():
        if value:
            ffmetadata.append(f"{key}={_escape_ffmetadata(str(value))}")

    chapters_by_base = {c["filename_base"]: c for c in metadata["chapters"]}
    current_ms = 0
    file_lines = []
    for wav_path in wav_paths:
        duration = _wav_duration_ms(wav_path)
        c = chapters_by_base.get(wav_path.stem)
        title = c["title"] if c else wav_path.stem
        ffmetadata += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={current_ms}",
            f"END={current_ms + duration}",
            f"title={_escape_ffmetadata(title)}",
        ]
        file_lines.append(f"file '{wav_path.resolve()}'")
        current_ms += duration

    meta_path = output_path.parent / "ffmetadata.txt"
    list_path = output_path.parent / "files.txt"
    meta_path.write_text("\n".join(ffmetadata), encoding="utf-8")
    list_path.write_text("\n".join(file_lines), encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
    input_idx = 1
    map_video = None
    if cover_path and cover_path.exists():
        cmd += ["-i", str(cover_path)]
        map_video = f"{input_idx}:v"
        input_idx += 1
    cmd += ["-i", str(meta_path)]
    cmd += ["-map_metadata", str(input_idx), "-map", "0:a"]
    if map_video:
        cmd += ["-map", map_video, "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, str(output_path)]

    _log(on_progress, f"building {output_path.name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        _log(on_progress, f"ffmpeg failed: {e.stderr.decode()}")
        raise
    finally:
        meta_path.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)

    return output_path


def synthesize_book(
    plan: BookPlan,
    engine: Qwen3TTSEngine | Qwen3VoiceDesignEngine,
    voice_kwargs: dict,
    run_state: dict,
    chunks_total: int,
    book_index: int,
    books_total: int,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """synthesize plan's selected chapters and build its m4b. resumable at both
    the chapter level (skips chapters whose wav already exists) and book level
    (caller should skip calling this at all if plan.output_path already exists).
    """
    plan.wav_dir.mkdir(parents=True, exist_ok=True)

    _emit(on_progress, {
        "event": "book_start",
        "book": plan.metadata["title"],
        "book_index": book_index,
        "books_total": books_total,
        "chapters_total": len(plan.selected_chapters),
    })

    wav_paths = []
    for chapter_num, c in enumerate(plan.selected_chapters, start=1):
        wav_path = plan.wav_dir / f"{c['filename_base']}.wav"
        chunks = plan.chapter_chunks[c["index"]]

        if wav_path.exists():
            wav_paths.append(wav_path)
            run_state["chunks_done"] += len(chunks)
            _emit(on_progress, {
                "event": "chapter_skip",
                "book": plan.metadata["title"],
                "chapter_index": c["index"],
                "chapter_title": c["title"],
                "chapter_num": chapter_num,
                "chapters_total": len(plan.selected_chapters),
            })
            continue

        _emit(on_progress, {
            "event": "chapter_start",
            "book": plan.metadata["title"],
            "chapter_index": c["index"],
            "chapter_title": c["title"],
            "chapter_num": chapter_num,
            "chapters_total": len(plan.selected_chapters),
            "chunks_in_chapter": len(chunks),
        })

        chunk_audio: list[np.ndarray] = []
        sr = SAMPLE_RATE
        for batch_start in range(0, len(chunks), engine.batch_size):
            batch = chunks[batch_start : batch_start + engine.batch_size]
            wavs, sr = engine.synthesize_batch(batch, **voice_kwargs)
            chunk_audio.extend(wavs)
            run_state["chunks_done"] += len(batch)

            elapsed = time.monotonic() - run_state["start_time"]
            rate = run_state["chunks_done"] / elapsed if elapsed > 0 else 0.0
            remaining = max(chunks_total - run_state["chunks_done"], 0)
            eta = remaining / rate if rate > 0 else None

            _emit(on_progress, {
                "event": "chunk_progress",
                "book": plan.metadata["title"],
                "chapter_index": c["index"],
                "chapter_title": c["title"],
                "chunks_done_chapter": len(chunk_audio),
                "chunks_in_chapter": len(chunks),
                "chunks_done": run_state["chunks_done"],
                "chunks_total": chunks_total,
                "elapsed_sec": elapsed,
                "chunks_per_sec": rate,
                "eta_sec": eta,
            })

        audio = concatenate_audio(chunk_audio, sr, PARAGRAPH_PAUSE_MS)
        sf.write(str(wav_path), audio, sr)
        wav_paths.append(wav_path)

        # release cached allocator blocks between chapters so reserved-but-unused
        # memory doesn't fragment and accumulate over a long (e.g. 270-chapter) run
        if engine._torch.cuda.is_available():
            engine._torch.cuda.empty_cache()

        _emit(on_progress, {
            "event": "chapter_done",
            "book": plan.metadata["title"],
            "chapter_index": c["index"],
            "chapter_title": c["title"],
            "chapter_num": chapter_num,
            "chapters_total": len(plan.selected_chapters),
        })

    cover_path = plan.extract_dir / COVER_FILE
    cover_path = cover_path if cover_path.exists() else None
    build_m4b(plan.metadata, wav_paths, cover_path, plan.output_path, on_progress=on_progress)

    _emit(on_progress, {
        "event": "book_done",
        "book": plan.metadata["title"],
        "book_index": book_index,
        "books_total": books_total,
        "output_path": str(plan.output_path),
    })
    return plan.output_path


def run_batch(
    content_dir: str | Path = "/content",
    work_dir: str | Path = "/content/zipcast_work",
    epub_paths: list[str | Path] | None = None,
    chapters: str = "all",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
    voice_mode: str = "preset",
    speaker: str = DEFAULT_SPEAKER,
    voice_description: str = "",
    model_name: str = DEFAULT_MODEL,
    design_model_name: str = VOICE_DESIGN_MODEL,
    engine: "Qwen3TTSEngine | Qwen3VoiceDesignEngine | None" = None,
    on_progress: ProgressCallback | None = None,
) -> list[Path]:
    """find every .epub in content_dir (or use epub_paths) and turn each into an
    .m4b, one after another, reporting progress via on_progress(event_dict).

    voice_mode "preset" uses a built-in named speaker (e.g. "ryan") via the
    CustomVoice model; voice_mode "design" narrates with a free-form voice
    description via the VoiceDesign model instead.
    """
    content_dir = Path(content_dir)
    work_dir = Path(work_dir)

    paths = [Path(p) for p in epub_paths] if epub_paths else find_epubs(content_dir)
    _log(on_progress, f"found {len(paths)} epub file(s): {', '.join(p.name for p in paths)}")

    # planning phase: parse + chunk everything upfront (CPU-only, cheap) so the
    # total chunk count -- and therefore ETA -- is accurate from the first batch
    _emit(on_progress, {"event": "planning", "books_total": len(paths)})
    plans: list[BookPlan] = []
    chunks_total = 0
    for epub_path in paths:
        plan = plan_book(epub_path, content_dir, work_dir, chapters_spec=chapters, chunk_chars=chunk_chars)
        if plan.output_path.exists():
            _log(on_progress, f"skip (already built): {plan.output_path.name}")
            continue
        plans.append(plan)
        chunks_total += plan.total_chunks

    _emit(on_progress, {"event": "plan_ready", "books_total": len(plans), "chunks_total": chunks_total})

    if not plans:
        _emit(on_progress, {"event": "all_done", "outputs": []})
        return []

    if engine is not None:
        # reuse a pre-loaded engine (e.g. preloaded before the GUI/tunnel even
        # started) instead of paying the multi-minute load cost on every job
        if isinstance(engine, Qwen3TTSEngine):
            engine.batch_size = max(1, batch_size)
        voice_kwargs = {"instruct": voice_description} if voice_mode == "design" else {"speaker": speaker}
    elif voice_mode == "design":
        engine = Qwen3VoiceDesignEngine(model_name=design_model_name, device=device, on_progress=on_progress)
        voice_kwargs = {"instruct": voice_description}
    else:
        engine = Qwen3TTSEngine(
            model_name=model_name, device=device, batch_size=batch_size, on_progress=on_progress
        )
        voice_kwargs = {"speaker": speaker}

    run_state = {"chunks_done": 0, "start_time": time.monotonic()}
    outputs = []
    for book_index, plan in enumerate(plans, start=1):
        output_path = synthesize_book(
            plan, engine, voice_kwargs, run_state, chunks_total, book_index, len(plans), on_progress=on_progress
        )
        outputs.append(output_path)

    # no _log() here: "all_done" already carries this info, and every consumer
    # (browser GUI, notebook mirror loop) renders it from the structured event --
    # logging it too would just print the same line a second time
    _emit(on_progress, {"event": "all_done", "outputs": [str(p) for p in outputs]})
    return outputs
