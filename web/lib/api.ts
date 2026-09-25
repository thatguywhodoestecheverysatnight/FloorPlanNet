"use client";
import type { Geometry, Segmentation } from "./types";

export { DEFAULT_API } from "./config";

export interface RemoteResult {
  geometry: Geometry;
  seg: Segmentation;
}

async function decodeMask(b64: string): Promise<{ label: Uint8Array; width: number; height: number }> {
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  const label = new Uint8Array(c.width * c.height);
  for (let i = 0; i < label.length; i++) label[i] = px[4 * i];
  return { label, width: c.width, height: c.height };
}

/** Full-size model served by the FastAPI/Docker backend (serve/app.py). */
export async function predictRemote(apiUrl: string, file: Blob, opts: { maxSide: number; tta: boolean }): Promise<RemoteResult> {
  const fd = new FormData();
  fd.append("file", file, "plan.png");
  fd.append("max_side", String(opts.maxSide));
  fd.append("tta", String(opts.tta));
  const t0 = performance.now();
  const r = await fetch(`${apiUrl.replace(/\/$/, "")}/api/predict`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`API ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  const m = await decodeMask(body.mask_png);
  return {
    geometry: body.geometry as Geometry,
    seg: { ...m, scale: body.mask_scale, meanConfidence: body.mean_confidence, ms: performance.now() - t0 },
  };
}

export async function apiHealth(apiUrl: string): Promise<boolean> {
  try {
    const r = await fetch(`${apiUrl.replace(/\/$/, "")}/health`, { signal: AbortSignal.timeout(4000) });
    return r.ok && (await r.json()).status === "ok";
  } catch {
    return false;
  }
}
