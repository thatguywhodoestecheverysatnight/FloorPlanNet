/**
 * Runtime asset locations. The Next.js build serves everything from the site root;
 * the standalone bundle (scripts/build-standalone.mjs) sets window.__FPN__ to relative paths.
 */
type FpnGlobal = { assetBase?: string; ortBase?: string; modelUrl?: string; apiUrl?: string };
const g: FpnGlobal = typeof window !== "undefined" ? ((window as unknown as { __FPN__?: FpnGlobal }).__FPN__ ?? {}) : {};

export const ASSET_BASE = g.assetBase ?? "/";
export const ORT_BASE = g.ortBase ?? `${ASSET_BASE}ort/`;
export const MODEL_URL = g.modelUrl || process.env.NEXT_PUBLIC_MODEL_URL || `${ASSET_BASE}models/floorplannet_lite.onnx`;
export const MODEL_CARD_URL = `${ASSET_BASE}models/model-card.json`;
export const DEFAULT_API = g.apiUrl ?? process.env.NEXT_PUBLIC_API_URL ?? "";
export const asset = (p: string) => `${ASSET_BASE}${p.replace(/^\//, "")}`;
