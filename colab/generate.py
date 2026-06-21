"""Colab-side audiobook generation: find+validate the zip from `zipcast`, run
Qwen3-TTS over each chapter, and assemble the result into a single m4b with
chapter markers and cover art.

Designed to be imported from the companion notebook after cloning this repo:

    from colab.generate import run_pipeline
    run_pipeline()
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
import zipfile
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


def find_zip(content_dir: str | Path = "/content") -> Path:
    """locate a zip file in content_dir, preferring the most recently uploaded one."""
    content_dir = Path(content_dir)
    candidates = sorted(content_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"no .zip file found in {content_dir}. upload the zip produced by `zipcast` "
            "(drag it into the Colab Files panel on the left) and re-run this cell."
        )
    zip_path = candidates[0]
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        print(f"found {len(candidates)} zip files ({names}); using most recent: {zip_path.name}")
    return zip_path


def validate_zip(zip_path: Path) -> None:
    """raise a clear error if zip_path isn't a valid, non-corrupt zipcast bundle."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid zip file")
    with zipfile.ZipFile(zip_path) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise ValueError(f"{zip_path} is corrupt (bad member: {bad_member})")
        names = zf.namelist()
        if METADATA_FILE not in names:
            raise ValueError(f"{zip_path} has no {METADATA_FILE} -- this isn't a zipcast bundle")
        if not any(n.endswith(".txt") for n in names):
            raise ValueError(f"{zip_path} contains no chapter .txt files")


def extract_zip(zip_path: Path, extract_dir: Path) -> dict:
    """extract zip_path into extract_dir and return its parsed metadata.json."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return json.loads((extract_dir / METADATA_FILE).read_text(encoding="utf-8"))


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


class Qwen3TTSEngine:
    """thin wrapper around the `qwen-tts` package for chapter-by-chapter synthesis."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        import torch
        from qwen_tts import Qwen3TTSModel  # type: ignore

        self._torch = torch
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        print(f"loading {model_name} on {device} ({dtype})... this can take a few minutes.")
        self.model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=dtype, attn_implementation="sdpa"
        )
        print("model loaded.")

    def synthesize(self, text_chunks: list[str], speaker: str) -> tuple[np.ndarray, int]:
        with self._torch.inference_mode():
            wavs, sr = self.model.generate_custom_voice(
                text=text_chunks, speaker=speaker, instruct="", max_new_tokens=2048
            )
        if isinstance(wavs, np.ndarray):
            wavs = [wavs]
        return concatenate_audio(list(wavs), sr, PARAGRAPH_PAUSE_MS), sr


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


def run_pipeline(
    content_dir: str | Path = "/content",
    work_dir: str | Path = "/content/zipcast_work",
    speaker: str = DEFAULT_SPEAKER,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
) -> Path:
    """end-to-end: find zip -> validate -> extract -> synthesize -> build m4b."""
    content_dir = Path(content_dir)
    work_dir = Path(work_dir)
    extract_dir = work_dir / "extract"
    wav_dir = work_dir / "wav"

    zip_path = find_zip(content_dir)
    validate_zip(zip_path)
    print(f"using zip: {zip_path}")

    metadata = extract_zip(zip_path, extract_dir)
    print(f"book: {metadata['title']} by {metadata['author']} ({len(metadata['chapters'])} chapters)")

    engine = Qwen3TTSEngine(model_name=model_name, device=device)
    wav_paths = synthesize_chapters(metadata, extract_dir, wav_dir, engine, speaker=speaker)

    cover_path = extract_dir / COVER_FILE
    cover_path = cover_path if cover_path.exists() else None

    safe_title = "".join(c for c in metadata["title"] if c.isalnum() or c in " -_").strip() or "audiobook"
    output_path = content_dir / f"{safe_title}.m4b"
    build_m4b(metadata, wav_paths, cover_path, output_path)
    print(f"\ndone: {output_path}")
    return output_path
