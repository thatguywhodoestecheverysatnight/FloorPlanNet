import { Geometry, Pt } from "./types";

const f1 = (v: number) => (Math.round(v * 10) / 10).toString();
const pts = (p: Pt[]) => p.map(([x, y]) => `${f1(x)},${f1(y)}`).join(" ");
const esc = (s: string) => s.replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c]!);

export function toSVG(g: Geometry, opts: { labels?: boolean } = { labels: true }): string {
  const out = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${g.width} ${g.height}" width="${g.width}" height="${g.height}">`,
    `<rect width="100%" height="100%" fill="#ffffff"/>`,
    `<g id="rooms" fill="#81c7eb" fill-opacity="0.45" stroke="#2b7bb9" stroke-width="1">`,
    ...g.rooms.map((r) => `<polygon data-id="${r.id}" data-label="${esc(r.label)}" points="${pts(r.polygon)}"/>`),
    `</g><g id="walls" stroke="#262626" stroke-linecap="square">`,
    ...g.walls.map(
      (w) => `<line data-id="${w.id}" x1="${f1(w.p1[0])}" y1="${f1(w.p1[1])}" x2="${f1(w.p2[0])}" y2="${f1(w.p2[1])}" stroke-width="${f1(Math.max(1, w.thickness))}"/>`,
    ),
    `</g><g id="openings">`,
    ...g.openings.map((o) => {
      const c = o.kind === "door" ? "#ec7063" : "#58d68d";
      return `<polygon data-id="${o.id}" data-kind="${o.kind}" points="${pts(o.polygon)}" fill="${c}" stroke="${c}"/>`;
    }),
    `</g>`,
  ];
  if (opts.labels) {
    out.push(`<g id="labels" font-family="sans-serif" font-size="${Math.max(9, g.width / 70).toFixed(0)}" text-anchor="middle" fill="#1b3a57">`);
    for (const r of g.rooms) out.push(`<text x="${f1(r.centroid[0])}" y="${f1(r.centroid[1])}">${esc(r.label)}</text>`);
    out.push(`</g>`);
  }
  out.push(`</svg>`);
  return out.join("\n");
}

export function toGeoJSON(g: Geometry) {
  const ring = (p: Pt[]) => {
    const r = p.map(([x, y]) => [x, y]);
    return r.length && (r[0][0] !== r[r.length - 1][0] || r[0][1] !== r[r.length - 1][1]) ? [...r, r[0]] : r;
  };
  return {
    type: "FeatureCollection",
    properties: { width: g.width, height: g.height, crs: "image-pixels", px_per_meter: g.px_per_meter },
    features: [
      ...g.rooms.map((r) => ({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring(r.polygon), ...r.holes.map(ring)] },
        properties: { layer: "room", id: r.id, label: r.label, area_px: r.area_px, area_m2: r.area_m2 },
      })),
      ...g.walls.map((w) => ({
        type: "Feature",
        geometry: { type: "LineString", coordinates: [w.p1, w.p2] },
        properties: { layer: "wall", id: w.id, thickness: w.thickness, exterior: w.exterior },
      })),
      ...g.openings.map((o) => ({
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring(o.polygon)] },
        properties: { layer: o.kind, id: o.id, width: o.width, wall_id: o.wall_id },
      })),
    ],
  };
}

/** Minimal ASCII DXF (AC1009 / R12): opens in AutoCAD, LibreCAD, QCAD, Revit import, FreeCAD. */
export function toDXF(g: Geometry): string {
  const s = g.px_per_meter ? 1 / g.px_per_meter : 1;
  const X = (p: Pt) => p[0] * s;
  const Y = (p: Pt) => (g.height - p[1]) * s;
  const L: string[] = [];
  const add = (...kv: (string | number)[]) => {
    for (let i = 0; i < kv.length; i += 2) L.push(String(kv[i]), typeof kv[i + 1] === "number" ? (kv[i + 1] as number).toFixed(4) : String(kv[i + 1]));
  };
  const layers: [string, number][] = [["WALLS", 7], ["ROOMS", 5], ["DOORS", 1], ["WINDOWS", 3], ["LABELS", 8]];
  add(0, "SECTION", 2, "HEADER", 9, "$ACADVER", 1, "AC1009", 9, "$INSUNITS", 70, g.px_per_meter ? "6" : "0", 0, "ENDSEC");
  add(0, "SECTION", 2, "TABLES", 0, "TABLE", 2, "LAYER", 70, String(layers.length));
  for (const [n, c] of layers) add(0, "LAYER", 2, n, 70, "0", 62, String(c), 6, "CONTINUOUS");
  add(0, "ENDTAB", 0, "ENDSEC", 0, "SECTION", 2, "ENTITIES");
  const poly = (layer: string, p: Pt[], width = 0) => {
    add(0, "POLYLINE", 8, layer, 66, "1", 70, "1", 40, width * s, 41, width * s);
    for (const q of p) add(0, "VERTEX", 8, layer, 10, X(q), 20, Y(q), 30, 0);
    add(0, "SEQEND", 8, layer);
  };
  for (const w of g.walls) {
    add(0, "POLYLINE", 8, "WALLS", 66, "1", 70, "0", 40, w.thickness * s, 41, w.thickness * s);
    for (const q of [w.p1, w.p2]) add(0, "VERTEX", 8, "WALLS", 10, X(q), 20, Y(q), 30, 0);
    add(0, "SEQEND", 8, "WALLS");
  }
  for (const r of g.rooms) {
    poly("ROOMS", r.polygon);
    add(0, "TEXT", 8, "LABELS", 10, X(r.centroid), 20, Y(r.centroid), 30, 0, 40, 8 * s, 1, r.label);
  }
  for (const o of g.openings) poly(o.kind === "door" ? "DOORS" : "WINDOWS", o.polygon);
  add(0, "ENDSEC", 0, "EOF");
  return L.join("\n") + "\n";
}

type SaveNs = { save: (r: { filename: string; data: Blob | string }) => Promise<unknown> };
type ClaudeHost = { use?: (name: string) => Promise<SaveNs | null> };

/**
 * Save a file. Inside a claude.ai artifact the sandbox blocks plain downloads, so
 * the host's `downloads` capability is used (its extension allowlist has no .dxf
 * or .geojson, hence the .txt / .json suffixes there).
 */
export async function download(name: string, data: string | Blob, type = "application/octet-stream"): Promise<void> {
  const blob = typeof data === "string" ? new Blob([data], { type }) : data;
  const host = (window as unknown as { claude?: ClaudeHost }).claude;
  if (host?.use) {
    const dl = await host.use("downloads").catch(() => null);
    if (dl) {
      const filename = name.endsWith(".dxf") ? `${name}.txt` : name.endsWith(".geojson") ? `${name}.json` : name;
      await dl.save({ filename, data: blob }).catch((e: unknown) => console.warn("save declined", e));
      return;
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
