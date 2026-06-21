# zipcast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinishReddyK/zipcast-epub/blob/main/notebooks/colab_zipcast_qwen3tts.ipynb)

epub → audiobook `.m4b`, entirely on a free Colab GPU, via [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Usage

1. Open the notebook in Colab using the badge above, or directly:
   https://colab.research.google.com/github/VinishReddyK/zipcast-epub/blob/main/notebooks/colab_zipcast_qwen3tts.ipynb
2. Set **Runtime > Change runtime type > T4 GPU**.
3. Drag one or more `.epub` files into the Colab **Files** panel so they land in `/content`.
4. Run every cell top to bottom.

The notebook clones this repo (`REPO_URL` in the first code cell -- only change it if
you fork this repo elsewhere), finds every `.epub` in `/content`, and converts each one
to a `.m4b` in turn: parses chapters + metadata + cover art directly from the epub,
synthesizes each chapter with Qwen3-TTS, and muxes the result into a single `.m4b`
with chapter markers and cover art via ffmpeg. The last cell downloads every `.m4b`
produced.

Processing is resumable at two levels: re-running the conversion cell skips chapters
whose `.wav` already exists under `/content/zipcast_work/<book>/wav`, and skips books
whose `.m4b` already exists in `/content` -- so a dropped Colab session doesn't mean
starting over.

## project layout

- `zipcast/epub_parser.py` -- epub parsing (chapters, metadata, cover art). Imported
  directly by the notebook after cloning this repo; no install step needed in Colab.
- `colab/generate.py` -- the conversion pipeline: find epubs, parse, synthesize with
  Qwen3-TTS, build the m4b.
- `notebooks/colab_zipcast_qwen3tts.ipynb` -- the Colab notebook itself.

## requirements

- Colab: GPU runtime. `ffmpeg`, `qwen-tts`, `torch`, `ebooklib` etc. are all installed
  in-notebook from `colab/requirements.txt` -- no local setup required.
