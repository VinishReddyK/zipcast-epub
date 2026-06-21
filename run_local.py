"""Run zipcast locally, no Colab and no tunnel needed.

Starts the same Flask GUI used in Colab, but bound to localhost only and
pointed at a local content folder instead of /content. Picks up a CUDA GPU,
an Apple Silicon GPU (mps), or falls back to CPU.

    python3 run_local.py

Then open the printed localhost URL (it also tries to open it for you) and drop
.epub files into the printed content folder, or upload them from the page.
"""

from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = Path(os.environ.get("ZIPCAST_CONTENT_DIR", str(ROOT / "zipcast_content")))

# must be set before importing colab.webapp, which reads these at import time
os.environ.setdefault("ZIPCAST_CONTENT_DIR", str(CONTENT_DIR))
os.environ.setdefault("ZIPCAST_WORK_DIR", str(CONTENT_DIR / "zipcast_work"))
os.environ.setdefault("ZIPCAST_VOICES_DIR", str(CONTENT_DIR / "zipcast_voices"))

from colab import webapp  # noqa: E402


def _print_progress(event: Any) -> None:
    if isinstance(event, dict):
        if event.get("event") == "log":
            print(f"[zipcast] {event.get('message', '')}", flush=True)
        elif event.get("event") == "error":
            print(f"[zipcast] error: {event.get('message', '')}", flush=True)
        return
    print(event, flush=True)


def main() -> None:
    port = int(os.environ.get("ZIPCAST_PORT", "1234"))
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    device = webapp._device()
    print(f"[zipcast] device: {device}", flush=True)
    print(f"[zipcast] drop .epub files into: {CONTENT_DIR}", flush=True)

    url = f"http://localhost:{port}"
    print(f"[zipcast] ready: {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    if os.environ.get("ZIPCAST_PRELOAD", "").lower() in {"1", "true", "yes"}:
        print("[zipcast] preloading model in the background...", flush=True)
        threading.Thread(
            target=webapp.preload,
            kwargs={"device": device, "on_progress": _print_progress},
            daemon=True,
        ).start()
    else:
        print("[zipcast] model will load when conversion starts, so load logs show in the page.", flush=True)

    webapp.start_server(port=port)


if __name__ == "__main__":
    main()
