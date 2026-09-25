"use client";
/**
 * In-browser DeepLabV3+ inference with onnxruntime-web (WASM, SIMD, threads when
 * the page is cross-origin isolated). The runtime is self-hosted under /ort.
 */
import type { Segmentation } from "./types";
import { MODEL_CARD_URL, MODEL_URL, ORT_BASE } from "./config";

type Ort = typeof import("onnxruntime-web");
declare global {
  interface Window {
    ort?: Ort;
  }
}

export { MODEL_URL };
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortPromise: Promise<Ort> | null = null;
let sessionPromise: Promise<import("onnxruntime-web").InferenceSession> | null = null;

function loadOrt(): Promise<Ort> {
  if (ortPromise) return ortPromise;
  ortPromise = new Promise<Ort>((resolve, reject) => {
    if (window.ort) return resolve(window.ort);
    const s = document.createElement("script");
    s.src = `${ORT_BASE}ort.wasm.min.js`;
    s.async = true;
    s.onload = () => (window.ort ? resolve(window.ort) : reject(new Error("onnxruntime-web failed to initialise")));
    s.onerror = () => reject(new Error(`could not load ${ORT_BASE}ort.wasm.min.js (run npm install to copy the runtime)`));
    document.head.appendChild(s);
  }).then((ort) => {
    ort.env.wasm.wasmPaths = new URL(ORT_BASE, document.baseURI).href;
    const threads = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 1 : 1;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, threads) : 1;
    return ort;
  });
  ortPromise.catch(() => (ortPromise = null));
  return ortPromise;
}

export async function loadSession(onProgress?: (msg: string) => void) {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const ort = await loadOrt();
    onProgress?.("Downloading model weights");
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`model download failed (${res.status}) from ${MODEL_URL}`);
    // "*.b64.txt": base64-encoded weights, for static hosts that refuse to serve binary .onnx files
    const buf = MODEL_URL.endsWith(".b64.txt")
      ? Uint8Array.from(atob((await res.text()).trim()), (c) => c.charCodeAt(0))
      : new Uint8Array(await res.arrayBuffer());
    onProgress?.("Compiling model");
    return ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  })();
  sessionPromise.catch(() => (sessionPromise = null));
  return sessionPromise;
}

/** Resize so the long side equals ``maxSide``, pad to a multiple of 32 with white paper. */
function preprocess(img: CanvasImageSource & { width: number; height: number }, maxSide: number) {
  const iw = (img as HTMLImageElement).naturalWidth || img.width;
  const ih = (img as HTMLImageElement).naturalHeight || img.height;
  const scale = maxSide / Math.max(iw, ih);
  const w = Math.max(1, Math.round(iw * scale));
  const h = Math.max(1, Math.round(ih * scale));
  const W = Math.ceil(w / 32) * 32;
  const H = Math.ceil(h / 32) * 32;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, W, H).data;
  const plane = W * H;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = (px[4 * i] / 255 - MEAN[0]) / STD[0];
    data[plane + i] = (px[4 * i + 1] / 255 - MEAN[1]) / STD[1];
    data[2 * plane + i] = (px[4 * i + 2] / 255 - MEAN[2]) / STD[2];
  }
  return { data, W, H, w, h, scale };
}

export async function segment(
  img: HTMLImageElement,
  maxSide = 512,
  onProgress?: (msg: string) => void,
): Promise<Segmentation> {
  const ort = await loadOrt();
  const session = await loadSession(onProgress);
  onProgress?.("Running DeepLabV3+");
  await new Promise((r) => setTimeout(r, 16)); // let the UI paint the status first
  const t0 = performance.now();
  const { data, W, H, w, h, scale } = preprocess(img, maxSide);
  const input = new ort.Tensor("float32", data, [1, 3, H, W]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const logits = out[session.outputNames[0]].data as Float32Array;
  const K = logits.length / (W * H);
  const plane = W * H;
  const label = new Uint8Array(w * h);
  let confSum = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * W + x;
      let best = 0;
      let bv = -Infinity;
      for (let k = 0; k < K; k++) {
        const v = logits[k * plane + i];
        if (v > bv) {
          bv = v;
          best = k;
        }
      }
      let z = 0;
      for (let k = 0; k < K; k++) z += Math.exp(logits[k * plane + i] - bv);
      confSum += 1 / z;
      label[y * w + x] = best;
    }
  return { label, width: w, height: h, scale, meanConfidence: confSum / (w * h), ms: performance.now() - t0 };
}

export async function modelInfo(): Promise<Record<string, unknown> | null> {
  try {
    const r = await fetch(MODEL_CARD_URL);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}
