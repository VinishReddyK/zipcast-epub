"""command-line entry point: zipcast book.epub -o book.zip"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pack import pack_to_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zipcast",
        description="parse an epub into clean chapter text and pack it into a zip for Colab upload",
    )
    parser.add_argument("epub", type=Path, help="path to the source .epub file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="output zip path (default: <epub_stem>.zip)"
    )
    args = parser.parse_args()

    if not args.epub.exists():
        print(f"error: epub file not found: {args.epub}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.epub.with_suffix(".zip")

    print(f"parsing {args.epub}...")
    metadata = pack_to_zip(args.epub, output)

    print(f"\n{metadata['title']} by {metadata['author']}")
    print(f"{len(metadata['chapters'])} chapters:")
    for c in metadata["chapters"]:
        print(f"  {c['index']:02d}. {c['title']} ({c['word_count']} words)")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nwrote {output} ({size_mb:.1f} MB)")
    print("upload this zip into the Colab session's /content folder, then run the colab notebook.")


if __name__ == "__main__":
    main()
