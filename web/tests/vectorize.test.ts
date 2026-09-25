import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  components,
  distanceTransform,
  minAreaRect,
  orthogonalize,
  polyArea,
  rdp,
  skeletonPaths,
  summary,
  traceContour,
  vectorize,
} from "../lib/vectorize";
import { toDXF, toGeoJSON, toSVG } from "../lib/exporters";
import type { Pt } from "../lib/types";

const FIX = join(__dirname, "fixtures");
const meta = JSON.parse(readFileSync(join(FIX, "meta.json"), "utf8")) as {
  file: string;
  width: number;
  height: number;
  python: { rooms: number; walls: number; doors: number; windows: number; total_wall_length_px: number; total_room_area_px: number };
}[];

describe("primitives", () => {
  it("labels 4- and 8-connected components", () => {
    const W = 5, H = 5;
    const b = new Uint8Array(W * H);
    b[0] = 1; b[6] = 1; // diagonal pair
    b[24] = 1;
    expect(components(b, W, H, 4).count).toBe(3);
    expect(components(b, W, H, 8).count).toBe(2);
  });

  it("computes exact euclidean distance with border as background", () => {
    const W = 9, H = 9;
    const b = new Uint8Array(W * H).fill(1);
    const d = distanceTransform(b, W, H);
    expect(d[4 * W + 4]).toBeCloseTo(5, 5);
    expect(d[0]).toBeCloseTo(1, 5);
  });

  it("traces a rectangle contour", () => {
    const W = 10, H = 8;
    const b = new Uint8Array(W * H);
    for (let y = 2; y < 6; y++) for (let x = 3; x < 8; x++) b[y * W + x] = 1;
    const c = traceContour(b, W, H, 2 * W + 3);
    const poly = rdp(c, 0.5, true);
    expect(poly.length).toBe(4);
    expect(polyArea(poly)).toBeCloseTo(4 * 3, 5); // pixel-centre polygon
  });

  it("splits a skeleton cross into four arms", () => {
    const W = 21, H = 21;
    const s = new Uint8Array(W * H);
    for (let i = 2; i < 19; i++) {
      s[10 * W + i] = 1;
      s[i * W + 10] = 1;
    }
    expect(skeletonPaths(s, W, H).length).toBe(4);
  });

  it("rectifies near-orthogonal polygons", () => {
    const out = orthogonalize([[0, 0], [50, 1], [51, 40], [1, 41]] as Pt[], 10);
    out.forEach((a, i) => {
      const b = out[(i + 1) % out.length];
      expect(a[0] === b[0] || a[1] === b[1]).toBe(true);
    });
  });

  it("finds the minimum-area rectangle of a rotated box", () => {
    const pts: Pt[] = [];
    const t = Math.PI / 6;
    for (let u = -10; u <= 10; u++)
      for (let v = -3; v <= 3; v++) pts.push([100 + u * Math.cos(t) - v * Math.sin(t), 50 + u * Math.sin(t) + v * Math.cos(t)]);
    const r = minAreaRect(pts);
    expect(Math.max(r.w, r.h)).toBeCloseTo(20, 1);
    expect(Math.min(r.w, r.h)).toBeCloseTo(6, 1);
    expect(r.cx).toBeCloseTo(100, 3);
  });
});

describe("parity with the Python vectoriser", () => {
  for (const m of meta) {
    it(`matches ${m.file}`, () => {
      const buf = readFileSync(join(FIX, m.file));
      const label = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      const t0 = performance.now();
      const g = vectorize(label, m.width, m.height);
      const ms = performance.now() - t0;
      const s = summary(g);
      expect(s.rooms).toBe(m.python.rooms);
      expect(s.doors).toBe(m.python.doors);
      expect(s.windows).toBe(m.python.windows);
      expect(Math.abs(s.walls - m.python.walls)).toBeLessThanOrEqual(2);
      expect(Math.abs(s.roomArea - m.python.total_room_area_px) / m.python.total_room_area_px).toBeLessThan(0.03);
      expect(Math.abs(s.wallLength - m.python.total_wall_length_px) / m.python.total_wall_length_px).toBeLessThan(0.1);
      expect(g.openings.every((o) => o.wall_id !== null)).toBe(true);
      expect(ms).toBeLessThan(5000);
      expect(toSVG(g).startsWith("<svg")).toBe(true);
      expect(toGeoJSON(g).features.length).toBe(g.rooms.length + g.walls.length + g.openings.length);
      expect(toDXF(g)).toContain("EOF");
    });
  }

  it("rescales to the original image size", () => {
    const buf = readFileSync(join(FIX, meta[0].file));
    const label = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    const g = vectorize(label, 512, 512, {}, [1024, 1024]);
    expect(g.width).toBe(1024);
    const s1 = summary(vectorize(label, 512, 512));
    expect(summary(g).roomArea / s1.roomArea).toBeCloseTo(4, 1);
  });
});
