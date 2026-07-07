const MODEL_URL = "./model.onnx";
const METADATA_URL = "./model-metadata.json";

const input = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const emptyPreview = document.getElementById("empty-preview");
const statusEl = document.getElementById("status");
const verdictEl = document.getElementById("verdict");
const meterFill = document.getElementById("meter-fill");
const aiScoreEl = document.getElementById("ai-score");
const fakeProbEl = document.getElementById("fake-prob");
const realProbEl = document.getElementById("real-prob");

let session;
let metadata;

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadModel() {
  metadata = await fetch(METADATA_URL).then((res) => res.json());
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
  });
  setStatus("모델 준비 완료");
}

function imageToTensor(image) {
  const size = metadata.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, size, size);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const { mean, std } = metadata.normalization;
  const tensor = new Float32Array(1 * 3 * size * size);
  const plane = size * size;

  for (let i = 0; i < plane; i += 1) {
    const pixelIndex = i * 4;
    tensor[i] = (pixels[pixelIndex] / 255 - mean[0]) / std[0];
    tensor[plane + i] = (pixels[pixelIndex + 1] / 255 - mean[1]) / std[1];
    tensor[2 * plane + i] = (pixels[pixelIndex + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", tensor, [1, 3, size, size]);
}

function readImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ image, url });
    image.onerror = reject;
    image.src = url;
  });
}

async function classify(file) {
  if (!session) {
    setStatus("모델 로딩 중");
    return;
  }

  setStatus("분석 중");
  const { image, url } = await readImage(file);
  preview.src = url;
  preview.style.display = "block";
  emptyPreview.style.display = "none";

  const inputTensor = imageToTensor(image);
  const feeds = { [metadata.input_name]: inputTensor };
  const output = await session.run(feeds);
  const logits = Array.from(output[metadata.output_name].data);
  const probs = softmax(logits);
  const aiIndex = metadata.class_names.findIndex((name) => metadata.ai_class_names.includes(name));
  const realIndex = metadata.class_names.findIndex((name) => name === "REAL" || name === "RealArt");
  const aiProb = probs[aiIndex >= 0 ? aiIndex : 0];
  const realProb = realIndex >= 0 ? probs[realIndex] : 1 - aiProb;

  aiScoreEl.textContent = (aiProb * 100).toFixed(2);
  fakeProbEl.textContent = formatPercent(aiProb);
  realProbEl.textContent = formatPercent(realProb);
  meterFill.style.width = `${Math.round(aiProb * 100)}%`;
  verdictEl.textContent = aiProb >= 0.5 ? "AI 가능성 높음" : "REAL 가능성 높음";
  setStatus("분석 완료");
}

input.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) classify(file).catch((error) => {
    console.error(error);
    setStatus("분석 실패");
  });
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) classify(file).catch((error) => {
    console.error(error);
    setStatus("분석 실패");
  });
});

loadModel().catch((error) => {
  console.error(error);
  setStatus("모델 로딩 실패");
});
