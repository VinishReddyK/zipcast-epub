"""Colab-side audiobook generation: find every .epub dropped into /content, parse
each directly (no manual zip/upload step), run Qwen3-TTS over its chapters, and
assemble the result into a single m4b with chapter markers and cover art.

Designed to be imported from the companion notebook after cloning this repo:

    from colab.generate import run_batch
    run_batch()
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore

SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 500
PARAGRAPH_PAUSE_MS = 500
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKER = "ryan"
METADATA_FILE = "metadata.json"
COVER_FILE = "cover.jpg"


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


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
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


class Qwen3TTSEngine:
    """thin wrapper around the `qwen-tts` package for chapter-by-chapter synthesis."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda", batch_size: int = 4):
        import torch
        import transformers  # type: ignore
        from qwen_tts import Qwen3TTSModel  # type: ignore

        # silences the "Setting `pad_token_id` to `eos_token_id`..." notice that
        # transformers' generate() logs on every call -- harmless, but with chunks
        # now batched per-chapter this fires hundreds of times over a full book
        transformers.logging.set_verbosity_error()

        self._torch = torch
        # cap how many chunks go into one generate_custom_voice() call: peak GPU
        # memory scales with batch size x max_new_tokens, so batching a whole
        # chapter's chunks at once (the naive approach) OOMs on long chapters
        # even on a 14-16GiB T4.
        self.batch_size = max(1, batch_size)
        dtype = _select_dtype(torch, device)
        print(f"loading {model_name} on {device} ({dtype})... this can take a few minutes.")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        print("model loaded.")

    def synthesize(self, text_chunks: list[str], speaker: str) -> tuple[np.ndarray, int]:
        audio_batches: list[np.ndarray] = []
        sample_rate = SAMPLE_RATE
        for i in range(0, len(text_chunks), self.batch_size):
            batch = text_chunks[i : i + self.batch_size]
            with self._torch.inference_mode():
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=batch, speaker=speaker, instruct="", max_new_tokens=2048
                )
            if isinstance(wavs, np.ndarray):
                wavs = [wavs]
            audio_batches.extend(wavs)
        return concatenate_audio(audio_batches, sample_rate, PARAGRAPH_PAUSE_MS), sample_rate


def synthesize_chapters(
    metadata: dict,
    extract_dir: Path,
    wav_dir: Path,
    engine: Qwen3TTSEngine,
    speaker: str = DEFAULT_SPEAKER,
) -> list[Path]:
    """synthesize one wav per chapter, skipping chapters already rendered (resumable)."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = []
    chapters = sorted(metadata["chapters"], key=lambda c: c["index"])
    for c in chapters:
        txt_path = extract_dir / f"{c['filename_base']}.txt"
        wav_path = wav_dir / f"{c['filename_base']}.wav"
        if wav_path.exists():
            print(f"skip (already synthesized): {wav_path.name}")
            wav_paths.append(wav_path)
            continue

        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"synthesizing {txt_path.name} ({len(chunks)} chunks)...")
        audio, sr = engine.synthesize(chunks, speaker)
        sf.write(str(wav_path), audio, sr)
        wav_paths.append(wav_path)

        # release cached allocator blocks between chapters so reserved-but-unused
        # memory doesn't fragment and accumulate over a long (e.g. 270-chapter) run
        if engine._torch.cuda.is_available():
            engine._torch.cuda.empty_cache()
    return wav_paths


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

    print(f"building {output_path.name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr.decode()}")
        raise
    finally:
        meta_path.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)

    return output_path


def process_epub(
    epub_path: Path,
    content_dir: Path,
    work_dir: Path,
    engine: Qwen3TTSEngine,
    speaker: str = DEFAULT_SPEAKER,
) -> Path:
    """parse one epub, synthesize its chapters, and build its m4b. skips work already done."""
    book_dir = work_dir / epub_path.stem
    extract_dir = book_dir / "extract"
    wav_dir = book_dir / "wav"

    metadata = extract_book(epub_path, extract_dir)
    safe_title = "".join(c for c in metadata["title"] if c.isalnum() or c in " -_").strip() or epub_path.stem
    output_path = content_dir / f"{safe_title}.m4b"

    if output_path.exists():
        print(f"skip (already built): {output_path.name}")
        return output_path

    print(f"\n=== {metadata['title']} by {metadata['author']} ({len(metadata['chapters'])} chapters) ===")
    wav_paths = synthesize_chapters(metadata, extract_dir, wav_dir, engine, speaker=speaker)

    cover_path = extract_dir / COVER_FILE
    cover_path = cover_path if cover_path.exists() else None

    build_m4b(metadata, wav_paths, cover_path, output_path)
    print(f"done: {output_path}")
    return output_path


def run_batch(
    content_dir: str | Path = "/content",
    work_dir: str | Path = "/content/zipcast_work",
    speaker: str = DEFAULT_SPEAKER,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    batch_size: int = 4,
) -> list[Path]:
    """find every .epub in content_dir and turn each into an .m4b, one after another."""
    content_dir = Path(content_dir)
    work_dir = Path(work_dir)

    epub_paths = find_epubs(content_dir)
    print(f"found {len(epub_paths)} epub file(s): {', '.join(p.name for p in epub_paths)}")

    engine = Qwen3TTSEngine(model_name=model_name, device=device, batch_size=batch_size)

    outputs = [
        process_epub(epub_path, content_dir, work_dir, engine, speaker=speaker)
        for epub_path in epub_paths
    ]

    print(f"\nall done: {len(outputs)} audiobook(s) generated")
    return outputs
