const $ = (id) => document.getElementById(id);

let defaults = null;
let books = [];

function fmtEta(sec) {
  if (sec == null || !isFinite(sec)) return "eta: --";
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `eta: ${h}h ${m}m`;
  if (m > 0) return `eta: ${m}m ${s}s`;
  return `eta: ${s}s`;
}

function logLine(text, cls) {
  const log = $("log");
  const div = document.createElement("div");
  if (cls) div.className = cls;
  const t = new Date().toLocaleTimeString();
  div.textContent = `[${t}] ${text}`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function loadDefaults() {
  const res = await fetch("/api/defaults");
  defaults = await res.json();
  $("chunk-chars").value = defaults.chunk_chars;
  $("batch-size").value = defaults.batch_size;
  $("voice-test-text").placeholder = defaults.voice_test_text;

  const sel = $("speaker");
  sel.innerHTML = "";
  for (const s of defaults.speakers) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    if (s === defaults.default_speaker) opt.selected = true;
    sel.appendChild(opt);
  }
}

async function loadBooks() {
  const list = $("books-list");
  list.innerHTML = '<p class="muted">loading...</p>';
  const res = await fetch("/api/books");
  const data = await res.json();
  books = data.books;

  if (books.length === 0) {
    list.innerHTML = '<p class="muted">no .epub files yet -- upload one above.</p>';
    return;
  }

  list.innerHTML = "";
  for (const b of books) {
    const row = document.createElement("label");
    row.className = "book-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.dataset.filename = b.filename;
    row.appendChild(checkbox);

    const meta = document.createElement("div");
    meta.className = "book-meta";
    if (b.error) {
      meta.innerHTML = `<div class="book-title">${b.filename}</div><div class="book-sub error">${b.error}</div>`;
      checkbox.disabled = true;
      checkbox.checked = false;
    } else {
      meta.innerHTML = `<div class="book-title">${b.title}</div><div class="book-sub">${b.author} &middot; ${b.chapters} chapters</div>`;
    }
    row.appendChild(meta);
    list.appendChild(row);
  }
}

function selectedEpubs() {
  return Array.from(document.querySelectorAll("#books-list input[type=checkbox]:checked")).map(
    (el) => el.dataset.filename
  );
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.mode;
      $("voice-preset").classList.toggle("hidden", mode !== "preset");
      $("voice-design").classList.toggle("hidden", mode !== "design");
    });
  });
}

function currentVoiceMode() {
  return document.querySelector(".tab.active").dataset.mode;
}

async function testVoice() {
  const description = $("voice-description").value.trim();
  if (!description) {
    $("voice-test-status").textContent = "enter a description first";
    return;
  }
  const sampleText = $("voice-test-text").value.trim() || undefined;

  $("test-voice").disabled = true;
  $("voice-test-status").textContent = "loading voice-design model + generating (first run can take a few minutes)...";
  $("voice-test-audio").classList.add("hidden");

  try {
    const res = await fetch("/api/voice-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description, sample_text: sampleText }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `request failed (${res.status})`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = $("voice-test-audio");
    audio.src = url;
    audio.classList.remove("hidden");
    audio.play().catch(() => {});
    $("voice-test-status").textContent = "done.";
  } catch (e) {
    $("voice-test-status").textContent = `error: ${e.message}`;
  } finally {
    $("test-voice").disabled = false;
  }
}

function resetProgressUI() {
  $("progress-card").classList.remove("hidden");
  $("overall-fill").style.width = "0%";
  $("chapter-fill").style.width = "0%";
  $("overall-pct").textContent = "0%";
  $("overall-counts").textContent = "";
  $("overall-rate").textContent = "";
  $("overall-eta").textContent = "";
  $("current-book").textContent = "starting...";
  $("current-chapter").textContent = "—";
  $("outputs").innerHTML = "";
  $("log").innerHTML = "";
}

function handleEvent(ev) {
  switch (ev.event) {
    case "planning":
      logLine(`parsing ${ev.books_total} book(s)...`);
      break;
    case "plan_ready":
      logLine(`plan ready: ${ev.books_total} book(s) to convert, ${ev.chunks_total} chunks total`);
      break;
    case "book_start":
      $("current-book").textContent = `book ${ev.book_index}/${ev.books_total}: ${ev.book}`;
      logLine(`▶ starting "${ev.book}" (${ev.chapters_total} chapters)`);
      break;
    case "chapter_start":
      $("current-chapter").textContent = `chapter ${ev.chapter_num}/${ev.chapters_total}: ${ev.chapter_title} (${ev.chunks_in_chapter} chunks)`;
      $("chapter-fill").style.width = "0%";
      break;
    case "chapter_skip":
      logLine(`skip (already synthesized): ${ev.chapter_title}`);
      break;
    case "chunk_progress": {
      const pct = ev.chunks_total ? Math.min(100, (ev.chunks_done / ev.chunks_total) * 100) : 0;
      $("overall-fill").style.width = `${pct}%`;
      $("overall-pct").textContent = `${pct.toFixed(1)}%`;
      $("overall-counts").textContent = `${ev.chunks_done} / ${ev.chunks_total} chunks`;
      $("overall-rate").textContent = `${ev.chunks_per_sec.toFixed(2)} chunks/sec`;
      $("overall-eta").textContent = fmtEta(ev.eta_sec);

      const chapPct = ev.chunks_in_chapter ? Math.min(100, (ev.chunks_done_chapter / ev.chunks_in_chapter) * 100) : 0;
      $("chapter-fill").style.width = `${chapPct}%`;

      logLine(
        `  chunk ${ev.chunks_done_chapter}/${ev.chunks_in_chapter} of "${ev.chapter_title}" ` +
        `(${ev.chunks_done}/${ev.chunks_total} overall, ${ev.chunks_per_sec.toFixed(2)} chunks/sec, ${fmtEta(ev.eta_sec)})`
      );
      break;
    }
    case "log":
      logLine(ev.message);
      break;
    case "chapter_done":
      logLine(`✓ chapter ${ev.chapter_num}/${ev.chapters_total}: ${ev.chapter_title}`, "ok");
      break;
    case "book_done": {
      logLine(`✓✓ finished "${ev.book}"`, "ok");
      const a = document.createElement("a");
      a.href = `/api/download/${encodeURIComponent(ev.output_path.split("/").pop())}`;
      a.textContent = `⬇ download ${ev.output_path.split("/").pop()}`;
      a.setAttribute("download", "");
      $("outputs").appendChild(a);
      break;
    }
    case "all_done":
      logLine(`all done: ${ev.outputs.length} audiobook(s) generated`, "ok");
      $("current-chapter").textContent = "done.";
      $("start-job").disabled = false;
      break;
    case "error":
      logLine(`error: ${ev.message}`, "err");
      $("start-job").disabled = false;
      break;
    case "stream_end":
      $("start-job").disabled = false;
      break;
    default:
      break;
  }
}

async function startJob() {
  $("start-error").classList.add("hidden");
  const epubs = selectedEpubs();
  if (epubs.length === 0) {
    $("start-error").textContent = "select at least one book";
    $("start-error").classList.remove("hidden");
    return;
  }

  const mode = currentVoiceMode();
  const body = {
    epubs,
    chapters: $("chapters").value.trim() || "all",
    chunk_chars: parseInt($("chunk-chars").value, 10),
    batch_size: parseInt($("batch-size").value, 10),
    voice_mode: mode,
    speaker: $("speaker").value,
    voice_description: $("voice-description").value.trim(),
  };

  $("start-job").disabled = true;
  resetProgressUI();

  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);

    const events = new EventSource(`/api/jobs/${data.job_id}/events`);
    events.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      handleEvent(ev);
      if (ev.event === "stream_end") events.close();
    };
    events.onerror = () => {
      logLine("progress stream disconnected", "err");
      $("start-job").disabled = false;
      events.close();
    };
  } catch (e) {
    $("start-error").textContent = e.message;
    $("start-error").classList.remove("hidden");
    $("start-job").disabled = false;
  }
}

async function uploadBooks() {
  const input = $("upload-input");
  const files = input.files;
  if (!files || files.length === 0) {
    $("upload-status").textContent = "choose .epub file(s) first";
    return;
  }

  const form = new FormData();
  for (const f of files) form.append("files", f);

  $("upload-btn").disabled = true;
  $("upload-status").textContent = `uploading ${files.length} file(s)...`;

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `upload failed (${res.status})`);

    let msg = `uploaded ${data.saved.length} file(s).`;
    if (data.rejected && data.rejected.length) {
      msg += ` skipped: ${data.rejected.map((r) => `${r.filename} (${r.reason})`).join(", ")}`;
    }
    $("upload-status").textContent = msg;
    input.value = "";
    await loadBooks();
  } catch (e) {
    $("upload-status").textContent = `error: ${e.message}`;
  } finally {
    $("upload-btn").disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  loadDefaults();
  loadBooks();
  $("refresh-books").addEventListener("click", loadBooks);
  $("upload-btn").addEventListener("click", uploadBooks);
  $("test-voice").addEventListener("click", testVoice);
  $("start-job").addEventListener("click", startJob);
});
