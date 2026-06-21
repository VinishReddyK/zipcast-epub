# zipcast

epub → zip (locally) → audiobook `.m4b` (on a free Colab GPU, via Qwen3-TTS).

Split into two halves so the heavy TTS work never has to run on a weak laptop:

- **`zipcast`** (this package): parses an epub into clean chapter text + metadata
  + cover art, and packs it into a single zip. Runs anywhere, no GPU needed.
- **`colab/` + `notebooks/colab_zipcast_qwen3tts.ipynb`**: a Colab notebook that
  clones this repo, finds and validates the zip you uploaded into `/content`,
  runs [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) chapter by chapter, and
  muxes the result into a single `.m4b` with chapter markers and cover art.

## 1. Parse and zip the epub (locally)

```bash
pip install -e .
zipcast book.epub -o book.zip
```

This prints the parsed chapter list and writes `book.zip`, containing:

```
metadata.json       # title, author, language, chapter list
01_Chapter_One.txt
02_Chapter_Two.txt
...
cover.jpg           # if the epub has one
```

## 2. Generate the audiobook (Colab)

1. Push this repo to GitHub (or your git host of choice) so Colab can clone it.
2. Open `notebooks/colab_zipcast_qwen3tts.ipynb` in Colab and set the `REPO_URL`
   field in the first code cell to your repo's URL.
3. Set **Runtime > Change runtime type > T4 GPU**.
4. Drag `book.zip` into the Colab **Files** panel so it lands in `/content`.
5. Run every cell top to bottom. The last cell downloads the finished `.m4b`.

Synthesis is resumable: re-running the synthesis cell skips chapters whose `.wav`
already exists under `/content/zipcast_work/wav`, so a dropped Colab session
doesn't mean starting over.

## requirements

- local: Python 3.10+, `ebooklib`, `beautifulsoup4`, `lxml` (installed via `pip install -e .`)
- Colab: GPU runtime, `ffmpeg` (installed in-notebook), `qwen-tts` + `torch` (installed in-notebook)
