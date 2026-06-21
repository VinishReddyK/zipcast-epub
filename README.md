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
   - **Upload** one or more `.epub` files (or drag them into the Colab Files
     panel beforehand -- either works, the GUI lists whatever's in `/content`).
   - Pick which **chapters** to convert (`all`, `5`, `1-10`, `1,3,5-8`, ...).
   - Tune **chunk size** (characters of text per TTS call) and **batch size**
     (how many chunks get synthesized together -- higher is faster but uses
     more GPU memory; drop it if you hit a CUDA OOM on a free T4).
   - Pick a **voice**: either a built-in named speaker, or describe one in
     free text and click **test voice** to hear a preview before committing
     a whole book to it.
   - Click **Start conversion** and watch the live progress bar, ETA, and
     chunks/sec throughput. Each book gets a download link as soon as it's done.

Processing is resumable at two levels: re-running the conversion skips chapters
whose `.wav` already exists under `/content/zipcast_work/<book>/wav`, and skips
books whose `.m4b` already exists in `/content` -- so a dropped Colab session
doesn't mean starting over.

## project layout

- `zipcast/epub_parser.py` -- epub parsing (chapters, metadata, cover art).
- `colab/generate.py` -- the conversion pipeline: plan (parse + chunk every
  selected chapter upfront, for accurate progress/ETA), synthesize with
  Qwen3-TTS (named speaker or free-form voice description), build the m4b.
- `colab/webapp.py` -- the Flask backend: book listing, upload, job
  orchestration, live progress over Server-Sent Events, voice-description
  preview, file download.
- `colab/static/` -- the GUI itself (plain HTML/CSS/JS, no build step).
- `notebooks/colab_zipcast_qwen3tts.ipynb` -- clones this repo, installs
  dependencies, and launches the GUI + tunnel.

## requirements

- Colab: GPU runtime. `ffmpeg`, `qwen-tts`, `torch`, `flask`, `ebooklib` etc.
  are all installed in-notebook from `colab/requirements.txt` -- no local
  setup required.
