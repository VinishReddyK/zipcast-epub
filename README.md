# zipcast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinishReddyK/zipcast-epub/blob/main/notebooks/colab_zipcast_qwen3tts.ipynb)

epub &rarr; audiobook `.m4b`, entirely on a free Colab GPU, via [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
driven by a small web GUI instead of editing notebook cells.

## Usage

1. Open the notebook in Colab using the badge above, or directly:
   https://colab.research.google.com/github/VinishReddyK/zipcast-epub/blob/main/notebooks/colab_zipcast_qwen3tts.ipynb
2. Set **Runtime > Change runtime type > T4 GPU**.
3. Run every cell top to bottom. One cell **preloads the Qwen3-TTS model**
   before anything else starts, so the first conversion doesn't pay a
   multi-minute cold-load cost (the model stays cached for every job after
   that too). The last cell starts a local web app and opens a free Cloudflare
   "quick tunnel" to it (no account, no signup, no auth token) and prints a
   `https://*.trycloudflare.com` link.
4. Open that link. From the page:
   - Click **upload .epub(s)** -- it opens your file picker directly and
     uploads as soon as you choose files (or drag them into the Colab Files
     panel beforehand -- either works, the GUI lists whatever's in `/content`).
   - Pick which **chapters** to convert (`all`, `5`, `1-10`, `1,3,5-8`, ...).
     For books where the chapter numbering is positional within the file but
     the titles carry a series-wide number (e.g. volume 2 of a series whose
     first chapter is titled "Chapter 270"), use the chapter picker below the
     range field: search by title, then pick From/To by what's actually
     printed in the book -- it fills in the right range for you.
   - Tune **chunk size** (characters of text per TTS call) and **batch size**
     (how many chunks get synthesized together -- higher is faster but uses
     more GPU memory; drop it if you hit a CUDA OOM on a free T4).
   - Pick a **voice**: a built-in named speaker, a free-form text description
     (click **test voice** to preview it), or **clone one from an uploaded
     clip** -- a few seconds of clear speech plus its exact transcript (the
     model needs that to align the clone). **test clone** previews it the
     same way before you commit a whole book to it.
   - Click **Start conversion** and watch the live progress bar, ETA, and
     chunks/sec throughput (delivered over a WebSocket, so it updates as it
     happens rather than in laggy bursts). Each book gets a download link as
     soon as it's done. If you refresh the page mid-conversion, it reattaches
     to the running job and replays everything that's happened so far.

Processing is resumable at two levels: re-running the conversion skips chapters
whose `.wav` already exists under `/content/zipcast_work/<book>/wav`, and skips
books whose `.m4b` already exists in `/content` -- so a dropped Colab session
doesn't mean starting over.

## project layout

- `zipcast/epub_parser.py` -- epub parsing (chapters, metadata, cover art).
- `colab/generate.py` -- the conversion pipeline: plan (parse + chunk every
  selected chapter upfront, for accurate progress/ETA), synthesize with
  Qwen3-TTS (named speaker, free-form voice description, or a voice cloned
  from a reference clip), build the m4b.
- `colab/webapp.py` -- the Flask backend: book listing, epub + reference-clip
  upload (clips get normalized through ffmpeg on upload), job orchestration,
  live progress over WebSocket, per-book chapter listing (for the title-based
  range picker), voice preview (description or clone), file download.
- `colab/static/` -- the GUI itself (plain HTML/CSS/JS, no build step).
- `notebooks/colab_zipcast_qwen3tts.ipynb` -- clones this repo, installs
  dependencies, and launches the GUI + tunnel.

## requirements

- Colab: GPU runtime. `ffmpeg`, `qwen-tts`, `torch`, `flask`, `ebooklib` etc.
  are all installed in-notebook from `colab/requirements.txt` -- no local
  setup required.
