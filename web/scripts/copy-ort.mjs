// Copies the onnxruntime-web WASM runtime into /public/ort so it is served
// same-origin (works offline, behind strict CSPs, and with COOP/COEP threading).
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dst = join(root, "public", "ort");
if (!existsSync(src)) {
  console.warn("[copy-ort] onnxruntime-web not installed yet, skipping");
  process.exit(0);
}
mkdirSync(dst, { recursive: true });
const keep = (f) =>
  f === "ort.wasm.min.js" ||
  (f.startsWith("ort-wasm-simd-threaded") && !/jsep|jspi|asyncify|webgpu/.test(f) && /\.(wasm|mjs)$/.test(f));
const files = readdirSync(src).filter(keep);
for (const f of files) cpSync(join(src, f), join(dst, f));
console.log(`[copy-ort] copied ${files.length} files -> public/ort: ${files.join(", ")}`);
