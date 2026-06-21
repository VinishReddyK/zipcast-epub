"""pack a parsed epub into a single zip bundle for upload to Colab."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .epub_parser import parse_epub

METADATA_FILE = "metadata.json"
COVER_FILE = "cover.jpg"


def pack_to_zip(epub_path: Path, zip_path: Path) -> dict:
    """parse epub_path and write metadata.json + NN_Title.txt (+ cover.jpg) into zip_path.

    returns the metadata dict that was written.
    """
    book, cover_data = parse_epub(epub_path)
    if not book.chapters:
        raise ValueError(f"no chapters extracted from {epub_path} (epub may be malformed or DRM-protected)")

    metadata = book.to_metadata()

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(METADATA_FILE, json.dumps(metadata, indent=2, ensure_ascii=False))
        for chapter in book.chapters:
            zf.writestr(f"{chapter.filename_base}.txt", chapter.text)
        if cover_data:
            zf.writestr(COVER_FILE, cover_data)

    return metadata
