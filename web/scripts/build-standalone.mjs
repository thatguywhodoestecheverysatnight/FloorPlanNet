// Builds the app into dist-standalone/ as plain static files with relative paths:
//   index.html, app.js, app.css, ort/*, models/*, samples/*
// Useful for GitHub Pages, S3, itch-style hosts or any CDN without Next.js.
import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "dist-standalone");
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

await build({
  entryPoints: [join(root, "standalone", "main.tsx")],
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2020",
  jsx: "automatic",
  outfile: join(out, "app.js"),
  alias: { "@": root },
  define: {
    "process.env.NODE_ENV": '"production"',
    "process.env.NEXT_PUBLIC_MODEL_URL": '""',
    "process.env.NEXT_PUBLIC_API_URL": '""',
    "process.env.NEXT_PUBLIC_REPO_URL": JSON.stringify(process.env.NEXT_PUBLIC_REPO_URL || "https://github.com/"),
  },
  loader: { ".css": "css" },
  logLevel: "info",
});

for (const d of ["ort", "models", "samples"]) cpSync(join(root, "public", d), join(out, d), { recursive: true });
cpSync(join(root, "public", "favicon.svg"), join(out, "favicon.svg"));

const inline = process.argv.includes("--inline-css");
const css = readFileSync(join(out, "app.css"), "utf8");
writeFileSync(
  join(out, "index.html"),
  `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FloorPlanNet</title>
<link rel="icon" href="favicon.svg" />
${inline ? `<style>${css}</style>` : `<link rel="stylesheet" href="app.css" />`}
<script>window.__FPN__ = { assetBase: "./" };</script>
</head>
<body>
<div id="root"></div>
<script src="app.js"></script>
</body>
</html>
`,
);
console.log("standalone build ->", out);
