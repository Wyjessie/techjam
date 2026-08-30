const API_BASE = "http://localhost:8000";
const MAX_BATCH_SIZE = 30;

// ---- sample images (pre-loaded so the grid isn't empty on first load) —
// bundled locally under assets/samples/ (teammate-provided, from SID_Set /
// WildFake), ground truth + generator come from assets/samples/manifest.json
// rather than being hardcoded here. Filenames are deliberately neutral
// (sample_01.jpg, ...) so a screenshot doesn't leak which is which —
// see the manifest's own "note" field.
const SAMPLE_MANIFEST_URL = "assets/samples/manifest.json";

async function fetchAsFile(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const blob = await res.blob();
  return new File([blob], url.split("/").pop(), { type: blob.type || "image/jpeg" });
}

// ---- state ----
// Flat list, no generator/matrix concept. Each image carries its own ground
// truth, the AI model name if it's labeled AI, and — once judged — the
// model's verdict. In-memory only, the backend has no notion of any of it.
let images = [];
// { id, file, name, groundTruth: null|"authentic"|"ai_generated", modelName, result }

// "editing" (upload/label/tick grid) -> "judged" (spectrum, answers hidden)
// -> "revealed" (spectrum, tick/cross shown). Adding or removing an image
// invalidates a judged/revealed run, so it drops back to "editing".
let phase = "editing";

function uid() {
  return (crypto.randomUUID && crypto.randomUUID()) || String(Date.now() + Math.random());
}
function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}

// ---- element refs ----
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const errorEl = document.getElementById("error");
const imageGrid = document.getElementById("image-grid");
const judgeBtn = document.getElementById("judge-btn");
const judgeProgress = document.getElementById("judge-progress");
const judgeProgressFill = document.getElementById("judge-progress-fill");
const judgeProgressText = document.getElementById("judge-progress-text");
const spectrumView = document.getElementById("spectrum-view");
const spectrumReal = document.getElementById("spectrum-real");
const spectrumAi = document.getElementById("spectrum-ai");
const revealBtn = document.getElementById("reveal-btn");
const judgingStats = document.getElementById("judging-stats");

// ============ UPLOAD ============

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => addFiles(Array.from(e.dataTransfer.files || [])));
fileInput.addEventListener("change", (e) => {
  addFiles(Array.from(e.target.files || []));
  fileInput.value = "";
});

function addFiles(files) {
  const picked = files.filter((f) => f.type.startsWith("image/"));
  if (picked.length < files.length) {
    showError("Some files were skipped — only images are supported.");
  } else {
    hideError();
  }

  const room = Math.max(0, MAX_BATCH_SIZE - images.length);
  picked.slice(0, room).forEach((file) => {
    images.push({ id: uid(), file, name: file.name, groundTruth: null, modelName: "", result: null });
  });
  if (images.length >= MAX_BATCH_SIZE) showError(`Batch capped at ${MAX_BATCH_SIZE} images.`);
  phase = "editing"; // a changed image set invalidates any prior judged/revealed run
  render();
}

function removeImage(id) {
  images = images.filter((i) => i.id !== id);
  phase = "editing";
  render();
}

function setTruth(id, value) {
  const img = images.find((i) => i.id === id);
  if (!img) return;
  img.groundTruth = value;
  render();
}

function setModelName(id, value) {
  const img = images.find((i) => i.id === id);
  if (img) img.modelName = value; // no re-render — would drop input focus mid-type
}

// ============ RENDER ============

function render() {
  const editing = phase === "editing";
  imageGrid.classList.toggle("hidden", !editing);
  judgeBtn.classList.toggle("hidden", !editing);
  spectrumView.classList.toggle("hidden", editing);

  // Dropzone stays visible in every phase (you can always add more images),
  // but physically relocates to sit just below whichever "images" are on
  // screen and just above that phase's primary action button — Start
  // judging while editing, Reveal answer once judged.
  const anchor = editing ? judgeBtn : revealBtn;
  if (dropzone.nextElementSibling !== anchor) anchor.parentElement.insertBefore(dropzone, anchor);

  if (editing) renderGrid();
  else renderSpectrum();

  updateStats();
}

function renderGrid() {
  imageGrid.innerHTML = "";
  images.forEach((img, i) => {
    const card = buildCard(img);
    card.style.animationDelay = `${i * 30}ms`;
    imageGrid.appendChild(card);
  });
  judgeBtn.disabled = images.length === 0;
  judgeBtn.textContent = images.length > 0 ? `Start judging ${images.length} image${images.length > 1 ? "s" : ""}` : "Start judging";
}

function buildCard(img) {
  const card = el("div", "result-card");

  const frame = el("div", "frame");
  const imgEl = document.createElement("img");
  imgEl.src = URL.createObjectURL(img.file);
  frame.appendChild(imgEl);

  const remove = el("button", "remove");
  remove.textContent = "×";
  remove.title = "Remove";
  remove.addEventListener("click", (e) => {
    e.stopPropagation();
    removeImage(img.id);
  });
  frame.appendChild(remove);

  card.appendChild(frame);

  const body = el("div", "body");

  const name = el("div", "filename");
  name.textContent = img.name;
  body.appendChild(name);

  body.appendChild(buildTruthPick(img));

  if (img.groundTruth === "ai_generated") {
    const modelInput = document.createElement("input");
    modelInput.type = "text";
    modelInput.className = "model-input";
    modelInput.placeholder = "AI model (e.g. Midjourney)";
    modelInput.value = img.modelName;
    modelInput.addEventListener("input", (e) => setModelName(img.id, e.target.value));
    body.appendChild(modelInput);
  }

  card.appendChild(body);
  return card;
}

// ---- spectrum: judged images split Real <-> AI by the model's own verdict.
// Confidence used to also set distance-from-center, but got dropped when
// the model was returning ~100% on everything — that axis carried no
// signal, every item landed on the same two points. The teammate's update
// to Inference.py replaced the saturated sigmoid score with a genuine
// margin-to-threshold reliability index (see model/SQuaDE/Inference.py's
// confidence()), so it varies for real now — CSS `order` (below) restores
// distance-from-center from it: high confidence sits toward the outer
// edge of its side, low confidence sits near the divider. Layout is still
// a loose flex-wrap, not precise beeswarm-lane positions — good enough for
// "more confident -> more to the sides" without the lane math. ----

function infoLine(img) {
  if (img.groundTruth === "authentic") return "Real";
  if (img.groundTruth === "ai_generated") return img.modelName.trim() ? `AI · ${img.modelName.trim()}` : "AI";
  return "Unlabeled";
}

const SPECTRUM_MIN_SIZE = 52;
const SPECTRUM_SIZE_RANGE = 56; // thumbnails run 52-108px

// Stable per-image pseudo-random size — varied for visual texture, not to
// encode data, so it must not reshuffle on every re-render.
function sizeForId(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (Math.imul(h, 31) + id.charCodeAt(i)) >>> 0;
  return SPECTRUM_MIN_SIZE + (h % SPECTRUM_SIZE_RANGE);
}

function buildSpectrumItem(img, side) {
  const size = sizeForId(img.id);
  const item = el("div", "spectrum-item");

  // side="real" packs flex-start (its outer edge is the page's left edge),
  // side="ai" packs flex-end (its outer edge is the page's right edge) —
  // see .spectrum-side in style.css. `order` moves an item earlier/later
  // within that packing direction, so the mapping from confidence to a low
  // vs. high order value has to flip between the two sides to both mean
  // "toward the outer edge."
  const conf = typeof img.result.confidence === "number" ? img.result.confidence : 0.5;
  item.style.order = Math.round((side === "ai" ? conf : 1 - conf) * 1000);

  const thumb = el("div", "spectrum-thumb");
  thumb.style.width = thumb.style.height = `${size}px`;
  const imgEl = document.createElement("img");
  imgEl.src = URL.createObjectURL(img.file);
  thumb.appendChild(imgEl);

  if (phase === "revealed" && img.groundTruth) {
    const correct = img.result.label === img.groundTruth;
    const check = el("span", `spectrum-check revealed ${correct ? "correct" : "incorrect"}`);
    check.textContent = correct ? "✓" : "✗";
    check.style.fontSize = `${Math.round(size * 0.32)}px`;
    thumb.appendChild(check);
  }
  item.appendChild(thumb);

  // The label states the *ground truth*; combined with which side an item
  // sits on (the model's verdict), that's enough to eyeball correctness
  // without ever clicking reveal — so it stays invisible until then too.
  const info = el("div", "spectrum-info" + (phase === "revealed" ? " revealed" : ""));
  info.style.width = `${Math.max(size, 60)}px`;
  info.textContent = infoLine(img);
  item.appendChild(info);

  return item;
}

function renderSpectrum() {
  spectrumReal.innerHTML = "";
  spectrumAi.innerHTML = "";
  images
    .filter((i) => i.result && !i.result.error)
    .forEach((img) => {
      const isAi = img.result.label === "ai_generated";
      const side = isAi ? spectrumAi : spectrumReal;
      side.appendChild(buildSpectrumItem(img, isAi ? "ai" : "real"));
    });
}

function buildTruthPick(img) {
  const pick = el("div", "truth-pick");
  const selected = img.groundTruth === null ? "unknown" : img.groundTruth;
  [
    ["unknown", "?"],
    ["authentic", "Real"],
    ["ai_generated", "AI"],
  ].forEach(([value, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.className = value === selected ? `selected ${value}` : "";
    b.addEventListener("click", () => setTruth(img.id, value === "unknown" ? null : value));
    pick.appendChild(b);
  });
  return pick;
}

function updateStats() {
  const judged = images.filter((i) => i.result && !i.result.error);
  judgingStats.innerHTML = "";
  if (judged.length === 0) {
    judgingStats.classList.add("hidden");
    return;
  }

  // Confirmed/Missed/accuracy would leak the answer before "Reveal answer"
  // is clicked, so those tiles wait for phase === "revealed" too. No
  // "avg confidence" tile here — confidence is spent on spectrum layout
  // instead (distance from the divider, see buildSpectrumItem), so a
  // number here would just be a duplicate of what's already visible.
  const labeled = judged.filter((i) => i.groundTruth);
  const tiles = [{ value: judged.length, label: "judged" }];
  if (phase === "revealed" && labeled.length > 0) {
    const correct = labeled.filter((i) => i.result.label === i.groundTruth).length;
    tiles.push({ value: correct, label: "Confirmed", cls: "good" });
    tiles.push({ value: labeled.length - correct, label: "Missed", cls: "critical" });
    tiles.push({ value: `${Math.round((correct / labeled.length) * 100)}%`, label: "accuracy" });
  }

  tiles.forEach((t) => {
    const tile = el("div", "stat-tile" + (t.cls ? ` ${t.cls}` : ""));
    const value = el("div", "value");
    value.textContent = t.value;
    const label = el("div", "label");
    label.textContent = t.label;
    tile.append(value, label);
    judgingStats.appendChild(tile);
  });
  judgingStats.classList.remove("hidden");
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}
function hideError() {
  errorEl.classList.add("hidden");
}

// ============ JUDGE: one /api/predict call per image (not the batch
// endpoint) so the progress bar reflects images actually completed ============

judgeBtn.addEventListener("click", async () => {
  if (images.length === 0) return;
  hideError();
  judgeBtn.disabled = true;
  judgeBtn.textContent = "Judging…";
  showJudgeProgress(0, images.length);

  try {
    let done = 0;
    await Promise.all(
      images.map(async (img) => {
        const formData = new FormData();
        formData.append("file", img.file, img.name);
        try {
          const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
          img.result = res.ok ? await res.json() : { error: (await res.json().catch(() => ({}))).detail || `Request failed (${res.status})` };
        } catch (e) {
          img.result = { error: e.message || "Network error" }; // one image's failure shouldn't sink the rest
        }
        showJudgeProgress(++done, images.length);
      })
    );
    phase = "judged";
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    judgeBtn.disabled = false;
    judgeBtn.textContent = `Start judging ${images.length} image${images.length > 1 ? "s" : ""}`;
    hideJudgeProgress();
    render();
  }
});

function showJudgeProgress(done, total) {
  judgeProgress.classList.remove("hidden");
  judgeProgressFill.style.width = `${total ? (done / total) * 100 : 0}%`;
  judgeProgressText.textContent = `${done} / ${total} judged`;
}
function hideJudgeProgress() {
  judgeProgress.classList.add("hidden");
}

revealBtn.addEventListener("click", () => {
  phase = "revealed";
  revealBtn.disabled = true;
  revealBtn.textContent = "Answers revealed";
  render();
});

// ---- how-it-works: autoplay opts out under reduced-motion ----
// The <video autoplay> attribute has no reduced-motion awareness of its
// own, so honor the preference in JS instead.
if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll("#how-it-works video[autoplay]").forEach((v) => {
    v.removeAttribute("autoplay");
    v.pause();
  });
}

// ---- seed the grid with a few pre-loaded images so it isn't empty on
// first load. Purely a starting point — each is removable and the
// dropzone still adds custom ones. Ground truth + generator name come from
// the manifest, not guessed from the (deliberately neutral) filename.
async function seedSampleImages() {
  let manifest;
  try {
    manifest = await (await fetch(SAMPLE_MANIFEST_URL)).json();
  } catch (e) {
    return; // no bundled samples available — grid just starts empty
  }

  for (const entry of manifest.samples || []) {
    try {
      const file = await fetchAsFile(`assets/samples/${entry.image}`);
      images.push({
        id: uid(),
        file,
        name: file.name,
        groundTruth: entry.label_name === "fake" ? "ai_generated" : "authentic",
        modelName: entry.generator || "",
        result: null,
      });
    } catch (e) {
      // one bad sample shouldn't block the rest of the grid
    }
  }
  render();
}

// ---- init ----
render();
seedSampleImages();
