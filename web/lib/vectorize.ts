/**
 * Browser port of floorplannet/vectorize/vectorizer.py.
 *
 * mask -> clean -> walls (skeleton graph, Douglas-Peucker, Manhattan snap,
 * collinear merge, junction closure) + rooms (contours, orthogonalisation) +
 * openings (min-area rectangles, host wall assignment).
 *
 * Pure TypeScript, no OpenCV.js (saves ~8 MB of WASM). Tolerances scale with
 * the median wall thickness, like the Python reference.
 */
import { DOOR, Geometry, Opening, Pt, ROOM, Room, WALL, WINDOW, Wall } from "./types";

export interface VectorizeConfig {
  minRoomArea: number;
  minOpeningArea: number;
  minSpeckle: number;
  rdpRoom: number;
  rdpWall: number;
  snapAngleDeg: number;
  mergeOffset: number;
  mergeGap: number;
  junctionTol: number;
  spurLen: number;
  minWallLen: number;
  splitRadius: number;
  manhattan: boolean;
  pxPerMeter: number | null;
}

export const DEFAULT_CONFIG: VectorizeConfig = {
  minRoomArea: 200,
  minOpeningArea: 10,
  minSpeckle: 12,
  rdpRoom: 0.35,
  rdpWall: 0.35,
  snapAngleDeg: 12,
  mergeOffset: 0.9,
  mergeGap: 1.2,
  junctionTol: 1.6,
  spurLen: 1.5,
  minWallLen: 1.5,
  splitRadius: 1.0,
  manhattan: true,
  pxPerMeter: null,
};

const r2 = (v: number) => Math.round(v * 100) / 100;

// ============================================================ connected components
export interface Components {
  labels: Int32Array;
  count: number;
  areas: number[];
}

export function components(bin: Uint8Array, W: number, H: number, conn: 4 | 8 = 4): Components {
  const labels = new Int32Array(W * H);
  const areas: number[] = [0];
  const stack = new Int32Array(W * H);
  let count = 0;
  for (let i = 0; i < W * H; i++) {
    if (!bin[i] || labels[i]) continue;
    count++;
    let area = 0;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = count;
    while (sp) {
      const p = stack[--sp];
      area++;
      const x = p % W;
      const y = (p / W) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if ((dx === 0 && dy === 0) || (conn === 4 && dx !== 0 && dy !== 0)) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= W) continue;
          const q = ny * W + nx;
          if (bin[q] && !labels[q]) {
            labels[q] = count;
            stack[sp++] = q;
          }
        }
      }
    }
    areas.push(area);
  }
  return { labels, count, areas };
}

/** Remove speckles below minSize and refill them from the nearest surviving pixel (BFS). */
export function cleanMask(label: Uint8Array, W: number, H: number, numClasses: number, minSize: number): Uint8Array {
  const out = label.slice();
  const invalid = new Uint8Array(W * H);
  const bin = new Uint8Array(W * H);
  for (let c = 0; c < numClasses; c++) {
    let any = false;
    for (let i = 0; i < W * H; i++) {
      bin[i] = label[i] === c ? 1 : 0;
      any ||= bin[i] === 1;
    }
    if (!any) continue;
    const cc = components(bin, W, H, 8);
    for (let i = 0; i < W * H; i++) if (cc.labels[i] && cc.areas[cc.labels[i]] < minSize) invalid[i] = 1;
  }
  const queue = new Int32Array(W * H);
  let qh = 0;
  let qt = 0;
  for (let i = 0; i < W * H; i++) if (!invalid[i]) queue[qt++] = i;
  if (qt === 0) return out;
  while (qh < qt) {
    const p = queue[qh++];
    const x = p % W;
    const y = (p / W) | 0;
    const nb = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, y > 0 ? p - W : -1, y < H - 1 ? p + W : -1];
    for (const q of nb) {
      if (q >= 0 && invalid[q]) {
        invalid[q] = 0;
        out[q] = out[p];
        queue[qt++] = q;
      }
    }
  }
  return out;
}

/**
 * Marker-based splitting: erode by ``radius`` to cut thin leaks through doorways,
 * label the surviving cores, then grow them back over the mask with a BFS.
 */
export function splitRegions(bin: Uint8Array, W: number, H: number, radius: number, minMarker = 4): Components {
  const r = Math.round(radius);
  if (r < 1) return components(bin, W, H, 4);
  // separable square erosion, outside counts as background
  const tmp = new Uint8Array(W * H);
  const core = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let v = 1;
      for (let d = -r; d <= r && v; d++) {
        const xx = x + d;
        v = xx < 0 || xx >= W ? 0 : bin[y * W + xx];
      }
      tmp[y * W + x] = v;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let v = 1;
      for (let d = -r; d <= r && v; d++) {
        const yy = y + d;
        v = yy < 0 || yy >= H ? 0 : tmp[yy * W + x];
      }
      core[y * W + x] = v;
    }
  const cc = components(core, W, H, 4);
  const remap = new Int32Array(cc.count + 1);
  let nid = 0;
  for (let i = 1; i <= cc.count; i++) if (cc.areas[i] >= minMarker) remap[i] = ++nid;
  const out = new Int32Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = remap[cc.labels[i]];
  // components of the full mask with no marker keep their own id
  const full = components(bin, W, H, 4);
  const hasMarker = new Uint8Array(full.count + 1);
  for (let i = 0; i < W * H; i++) if (out[i]) hasMarker[full.labels[i]] = 1;
  const own = new Int32Array(full.count + 1);
  for (let i = 1; i <= full.count; i++) if (!hasMarker[i]) own[i] = ++nid;
  for (let i = 0; i < W * H; i++) if (bin[i] && !out[i] && own[full.labels[i]]) out[i] = own[full.labels[i]];
  // BFS growth inside the mask
  const q = new Int32Array(W * H);
  let qh = 0, qt = 0;
  for (let i = 0; i < W * H; i++) if (out[i]) q[qt++] = i;
  while (qh < qt) {
    const p = q[qh++];
    const x = p % W;
    const nb = [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W];
    for (const n of nb) {
      if (n < 0 || n >= W * H || !bin[n] || out[n]) continue;
      out[n] = out[p];
      q[qt++] = n;
    }
  }
  const areas = new Array(nid + 1).fill(0);
  for (let i = 0; i < W * H; i++) if (out[i]) areas[out[i]]++;
  return { labels: out, count: nid, areas };
}

// ============================================================ distance transform
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/** Exact Euclidean distance (Felzenszwalb & Huttenlocher) from foreground to background; image border counts as background. */
export function distanceTransform(bin: Uint8Array, W: number, H: number): Float32Array {
  const PW = W + 2;
  const PH = H + 2;
  const INF = 1e12;
  const grid = new Float64Array(PW * PH);
  for (let y = 0; y < PH; y++)
    for (let x = 0; x < PW; x++) {
      const inside = x > 0 && y > 0 && x <= W && y <= H && bin[(y - 1) * W + (x - 1)];
      grid[y * PW + x] = inside ? INF : 0;
    }
  const n = Math.max(PW, PH);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  for (let x = 0; x < PW; x++) {
    for (let y = 0; y < PH; y++) f[y] = grid[y * PW + x];
    edt1d(f, PH, d, v, z);
    for (let y = 0; y < PH; y++) grid[y * PW + x] = d[y];
  }
  for (let y = 0; y < PH; y++) {
    for (let x = 0; x < PW; x++) f[x] = grid[y * PW + x];
    edt1d(f, PW, d, v, z);
    for (let x = 0; x < PW; x++) grid[y * PW + x] = d[x];
  }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = Math.sqrt(grid[(y + 1) * PW + x + 1]);
  return out;
}

// ============================================================ thinning
/** Zhang-Suen thinning to a 1-px 8-connected skeleton. */
export function skeletonize(bin: Uint8Array, W: number, H: number): Uint8Array {
  const img = bin.slice();
  const del: number[] = [];
  let changed = true;
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : img[y * W + x]);
  while (changed) {
    changed = false;
    for (let pass = 0; pass < 2; pass++) {
      del.length = 0;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          if (!img[y * W + x]) continue;
          const p2 = at(x, y - 1), p3 = at(x + 1, y - 1), p4 = at(x + 1, y), p5 = at(x + 1, y + 1);
          const p6 = at(x, y + 1), p7 = at(x - 1, y + 1), p8 = at(x - 1, y), p9 = at(x - 1, y - 1);
          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;
          const seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
          let A = 0;
          for (let i = 0; i < 8; i++) if (!seq[i] && seq[i + 1]) A++;
          if (A !== 1) continue;
          if (pass === 0 ? p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0 : p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0)
            del.push(y * W + x);
        }
      if (del.length) {
        changed = true;
        for (const i of del) img[i] = 0;
      }
    }
  }
  return img;
}

// ============================================================ skeleton graph
const N8: [number, number][] = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

function neighbours(sk: Uint8Array, W: number, H: number, y: number, x: number): number[] {
  const out: number[] = [];
  for (const [dy, dx] of N8) {
    const ny = y + dy;
    const nx = x + dx;
    if (ny < 0 || nx < 0 || ny >= H || nx >= W || !sk[ny * W + nx]) continue;
    if (dy !== 0 && dx !== 0 && (sk[(y + dy) * W + x] || sk[y * W + x + dx])) continue;
    out.push(ny * W + nx);
  }
  return out;
}

/** Node-to-node pixel paths (flat indices) of a skeleton, plus closed loops. */
export function skeletonPaths(sk: Uint8Array, W: number, H: number): number[][] {
  const nbrs = new Map<number, number[]>();
  for (let i = 0; i < W * H; i++) if (sk[i]) nbrs.set(i, neighbours(sk, W, H, (i / W) | 0, i % W));
  const nodes = new Set<number>();
  for (const [p, n] of nbrs) if (n.length !== 2) nodes.add(p);
  const seen = new Set<string>();
  const key = (a: number, b: number) => `${a},${b}`;
  const paths: number[][] = [];
  const walk = (start: number, first: number) => {
    const path = [start, first];
    seen.add(key(start, first));
    seen.add(key(first, start));
    let prev = start;
    let cur = first;
    while (!nodes.has(cur)) {
      const nxt = (nbrs.get(cur) || []).filter((n) => n !== prev && !seen.has(key(cur, n)));
      if (!nxt.length) break;
      const n = nxt[0];
      seen.add(key(cur, n));
      seen.add(key(n, cur));
      path.push(n);
      prev = cur;
      cur = n;
      if (cur === start) break;
    }
    return path;
  };
  for (const node of nodes) for (const n of nbrs.get(node) || []) if (!seen.has(key(node, n))) paths.push(walk(node, n));
  for (const [p, ns] of nbrs) {
    if (nodes.has(p)) continue;
    for (const n of ns)
      if (!seen.has(key(p, n))) {
        nodes.add(p);
        paths.push(walk(p, n));
        nodes.delete(p);
      }
  }
  return paths;
}

function pruneSpurs(paths: number[][], minLen: number): number[][] {
  const ends = new Map<number, number>();
  for (const p of paths) for (const e of [p[0], p[p.length - 1]]) ends.set(e, (ends.get(e) || 0) + 1);
  return paths.filter((p) => {
    const free = Number(ends.get(p[0]) === 1) + Number(ends.get(p[p.length - 1]) === 1);
    return !(free && p.length < minLen && paths.length > 1);
  });
}

// ============================================================ polylines
export function rdp(pts: Pt[], eps: number, closed = false): Pt[] {
  if (pts.length < 3) return pts.slice();
  if (closed) {
    // split at the two farthest points (same idea as cv2.approxPolyDP)
    let far = 0;
    let best = -1;
    for (let i = 1; i < pts.length; i++) {
      const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2;
      if (d > best) {
        best = d;
        far = i;
      }
    }
    const a = rdp(pts.slice(0, far + 1), eps);
    const b = rdp(pts.slice(far).concat([pts[0]]), eps);
    return a.slice(0, -1).concat(b.slice(0, -1));
  }
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack: [number, number][] = [[0, pts.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop()!;
    const [x1, y1] = pts[s];
    const [x2, y2] = pts[e];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const L = Math.hypot(dx, dy) || 1e-9;
    let idx = -1;
    let dmax = eps;
    for (let i = s + 1; i < e; i++) {
      const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + x2 * y1 - y2 * x1) / L;
      if (d > dmax) {
        dmax = d;
        idx = i;
      }
    }
    if (idx >= 0) {
      keep[idx] = 1;
      stack.push([s, idx], [idx, e]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

interface Seg { x1: number; y1: number; x2: number; y2: number; t: number }
const segLen = (s: Seg) => Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
const isH = (s: Seg) => Math.abs(s.y2 - s.y1) < 1e-6 && Math.abs(s.x2 - s.x1) > 0;
const isV = (s: Seg) => Math.abs(s.x2 - s.x1) < 1e-6 && Math.abs(s.y2 - s.y1) > 0;

function snapAxis(s: Seg, maxDeg: number): Seg {
  const ang = Math.abs((Math.atan2(s.y2 - s.y1, s.x2 - s.x1) * 180) / Math.PI) % 180;
  if (Math.min(ang, 180 - ang) <= maxDeg) {
    const y = (s.y1 + s.y2) / 2;
    return { x1: Math.min(s.x1, s.x2), y1: y, x2: Math.max(s.x1, s.x2), y2: y, t: s.t };
  }
  if (Math.abs(ang - 90) <= maxDeg) {
    const x = (s.x1 + s.x2) / 2;
    return { x1: x, y1: Math.min(s.y1, s.y2), x2: x, y2: Math.max(s.y1, s.y2), t: s.t };
  }
  return s;
}

function mergeCollinear(segs: Seg[], offTol: number, gapTol: number): Seg[] {
  const merge = (group: Seg[], horiz: boolean): Seg[] => {
    const fixed = (s: Seg) => (horiz ? s.y1 : s.x1);
    const items = group.slice().sort((a, b) => fixed(a) - fixed(b));
    const clusters: Seg[][] = [];
    for (const s of items) {
      const last = clusters[clusters.length - 1];
      if (last) {
        const w = last.reduce((a, q) => a + segLen(q), 0);
        const mean = last.reduce((a, q) => a + fixed(q) * segLen(q), 0) / Math.max(w, 1e-6);
        if (Math.abs(fixed(s) - mean) <= offTol) {
          last.push(s);
          continue;
        }
      }
      clusters.push([s]);
    }
    const out: Seg[] = [];
    for (const cl of clusters) {
      const spans = cl
        .map((q) => ({ lo: horiz ? q.x1 : q.y1, hi: horiz ? q.x2 : q.y2, q }))
        .sort((a, b) => a.lo - b.lo);
      let lo = spans[0].lo;
      let hi = spans[0].hi;
      let cur: Seg[] = [spans[0].q];
      const runs: { lo: number; hi: number; qs: Seg[] }[] = [];
      for (const sp of spans.slice(1)) {
        if (sp.lo <= hi + gapTol) {
          hi = Math.max(hi, sp.hi);
          cur.push(sp.q);
        } else {
          runs.push({ lo, hi, qs: cur });
          lo = sp.lo;
          hi = sp.hi;
          cur = [sp.q];
        }
      }
      runs.push({ lo, hi, qs: cur });
      for (const r of runs) {
        const w = r.qs.reduce((a, q) => a + segLen(q), 0) || 1;
        const c = r.qs.reduce((a, q) => a + fixed(q) * segLen(q), 0) / w;
        const t = r.qs.reduce((a, q) => a + q.t * segLen(q), 0) / w;
        out.push(horiz ? { x1: r.lo, y1: c, x2: r.hi, y2: c, t } : { x1: c, y1: r.lo, x2: c, y2: r.hi, t });
      }
    }
    return out;
  };
  return [...merge(segs.filter(isH), true), ...merge(segs.filter(isV), false), ...segs.filter((s) => !isH(s) && !isV(s))];
}

function closeJunctions(segs: Seg[], tol: number): Seg[] {
  const hs = segs.filter(isH);
  const vs = segs.filter(isV);
  for (let iter = 0; iter < 2; iter++) {
    for (const h of hs)
      for (const end of [0, 1]) {
        const ex = end === 0 ? h.x1 : h.x2;
        let best: Seg | null = null;
        let bd = tol;
        for (const v of vs)
          if (v.y1 - tol <= h.y1 && h.y1 <= v.y2 + tol) {
            const d = Math.abs(v.x1 - ex);
            if (d < bd) {
              best = v;
              bd = d;
            }
          }
        if (best) {
          if (end === 0) h.x1 = best.x1;
          else h.x2 = best.x1;
        }
      }
    for (const v of vs)
      for (const end of [0, 1]) {
        const ey = end === 0 ? v.y1 : v.y2;
        let best: Seg | null = null;
        let bd = tol;
        for (const h of hs)
          if (h.x1 - tol <= v.x1 && v.x1 <= h.x2 + tol) {
            const d = Math.abs(h.y1 - ey);
            if (d < bd) {
              best = h;
              bd = d;
            }
          }
        if (best) {
          if (end === 0) v.y1 = best.y1;
          else v.y2 = best.y1;
        }
      }
  }
  const others = segs.filter((s) => !isH(s) && !isV(s));
  const ends: Pt[] = [...hs, ...vs].flatMap((s) => [[s.x1, s.y1] as Pt, [s.x2, s.y2] as Pt]);
  for (const s of others) {
    for (const [kx, ky] of [["x1", "y1"], ["x2", "y2"]] as const) {
      let bi = -1;
      let bd = tol;
      ends.forEach(([ex, ey], i) => {
        const d = Math.hypot(s[kx] - ex, s[ky] - ey);
        if (d < bd) {
          bd = d;
          bi = i;
        }
      });
      if (bi >= 0) {
        s[kx] = ends[bi][0];
        s[ky] = ends[bi][1];
      }
    }
  }
  return [...hs, ...vs, ...others].filter((s) => segLen(s) > 0);
}

// ============================================================ polygons
export function orthogonalize(poly: Pt[], maxDeg: number): Pt[] {
  if (poly.length < 4) return poly;
  const kind = (a: Pt, b: Pt) => {
    const ang = Math.abs((Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI) % 180;
    return Math.min(ang, 180 - ang) <= maxDeg ? "H" : Math.abs(ang - 90) <= maxDeg ? "V" : "D";
  };
  let pts = poly.map((p) => [p[0], p[1]] as Pt);
  let n = pts.length;
  let kinds = pts.map((p, i) => kind(p, pts[(i + 1) % n]));
  const keep = pts.filter((_, i) => !(kinds[(i - 1 + n) % n] === kinds[i] && kinds[i] !== "D"));
  if (keep.length < 3) return poly;
  pts = keep;
  n = pts.length;
  kinds = pts.map((p, i) => kind(p, pts[(i + 1) % n]));
  const fixed = pts.map((a, i) => {
    const b = pts[(i + 1) % n];
    return kinds[i] === "H" ? (a[1] + b[1]) / 2 : kinds[i] === "V" ? (a[0] + b[0]) / 2 : NaN;
  });
  const out: Pt[] = pts.map((p, i) => {
    let [x, y] = p;
    for (const j of [(i - 1 + n) % n, i]) {
      if (kinds[j] === "H") y = fixed[j];
      else if (kinds[j] === "V") x = fixed[j];
    }
    return [x, y];
  });
  const dedup: Pt[] = [];
  for (const p of out) {
    const l = dedup[dedup.length - 1];
    if (!l || Math.hypot(p[0] - l[0], p[1] - l[1]) > 1e-6) dedup.push(p);
  }
  if (dedup.length > 2) {
    const a = dedup[0];
    const b = dedup[dedup.length - 1];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6) dedup.pop();
  }
  return dedup;
}

export function polyArea(p: Pt[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i];
    const [x2, y2] = p[(i + 1) % p.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

// Moore-neighbour contour tracing. Directions clockwise on screen (y down), starting West.
const MD: [number, number][] = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];
const dirIndex = (dx: number, dy: number) => MD.findIndex(([a, b]) => a === dx && b === dy);

export function traceContour(bin: Uint8Array, W: number, H: number, start: number): Pt[] {
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < W && y < H && bin[y * W + x] === 1;
  const sx = start % W;
  const sy = (start / W) | 0;
  const out: Pt[] = [[sx, sy]];
  let px = sx;
  let py = sy;
  let back = 0; // came "from" the west (raster-first pixel has a background west neighbour)
  let second: [number, number] | null = null;
  const maxSteps = 4 * W * H + 8;
  for (let step = 0; step < maxSteps; step++) {
    let found = false;
    for (let i = 1; i <= 8; i++) {
      const d = (back + i) % 8;
      const nx = px + MD[d][0];
      const ny = py + MD[d][1];
      if (inside(nx, ny)) {
        const pd = (back + i - 1) % 8;
        const cx = px + MD[pd][0];
        const cy = py + MD[pd][1];
        if (px === sx && py === sy && second && nx === second[0] && ny === second[1] && step > 0) return out.slice(0, -1);
        if (!second) second = [nx, ny];
        back = dirIndex(cx - nx, cy - ny);
        px = nx;
        py = ny;
        out.push([px, py]);
        found = true;
        break;
      }
    }
    if (!found) return out; // isolated pixel
  }
  return out;
}

// ============================================================ rotated rectangles
function convexHull(pts: Pt[]): Pt[] {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Pt[] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

export function minAreaRect(pts: Pt[]): { cx: number; cy: number; w: number; h: number; angle: number } {
  const hull = convexHull(pts);
  if (hull.length === 1) return { cx: hull[0][0], cy: hull[0][1], w: 0, h: 0, angle: 0 };
  let best = { area: Infinity, cx: 0, cy: 0, w: 0, h: 0, angle: 0 };
  const cand = hull.length === 2 ? [0] : hull.map((_, i) => i);
  for (const i of cand) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const th = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const ux = Math.cos(th), uy = Math.sin(th);
    let mnu = Infinity, mxu = -Infinity, mnv = Infinity, mxv = -Infinity;
    for (const [x, y] of hull) {
      const u = x * ux + y * uy;
      const v = -x * uy + y * ux;
      mnu = Math.min(mnu, u); mxu = Math.max(mxu, u);
      mnv = Math.min(mnv, v); mxv = Math.max(mxv, v);
    }
    const area = (mxu - mnu) * (mxv - mnv);
    if (area < best.area - 1e-9) {
      const cu = (mnu + mxu) / 2;
      const cv = (mnv + mxv) / 2;
      best = { area, cx: cu * ux - cv * uy, cy: cu * uy + cv * ux, w: mxu - mnu, h: mxv - mnv, angle: (th * 180) / Math.PI };
    }
  }
  return best;
}

function boxPoints(cx: number, cy: number, w: number, h: number, angleDeg: number): Pt[] {
  const t = (angleDeg * Math.PI) / 180;
  const ux = Math.cos(t), uy = Math.sin(t);
  const vx = -uy, vy = ux;
  const hw = w / 2, hh = h / 2;
  return [
    [cx - ux * hw - vx * hh, cy - uy * hw - vy * hh],
    [cx + ux * hw - vx * hh, cy + uy * hw - vy * hh],
    [cx + ux * hw + vx * hh, cy + uy * hw + vy * hh],
    [cx - ux * hw + vx * hh, cy - uy * hw + vy * hh],
  ].map(([x, y]) => [r2(x), r2(y)] as Pt);
}

// ============================================================ main
function median(a: number[]): number {
  if (!a.length) return 0;
  const s = a.slice().sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function morphClose(bin: Uint8Array, W: number, H: number, k: number): Uint8Array {
  if (k <= 1) return bin;
  const r = Math.floor(k / 2);
  const pass = (src: Uint8Array, dilate: boolean) => {
    const tmp = new Uint8Array(W * H);
    const out = new Uint8Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = dilate ? 0 : 1;
        for (let d = -r; d <= r; d++) {
          const xx = Math.min(W - 1, Math.max(0, x + d));
          v = dilate ? v | src[y * W + xx] : v & src[y * W + xx];
        }
        tmp[y * W + x] = v;
      }
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = dilate ? 0 : 1;
        for (let d = -r; d <= r; d++) {
          const yy = Math.min(H - 1, Math.max(0, y + d));
          v = dilate ? v | tmp[yy * W + x] : v & tmp[yy * W + x];
        }
        out[y * W + x] = v;
      }
    return out;
  };
  return pass(pass(bin, true), false);
}

export function vectorize(
  labelIn: Uint8Array,
  W: number,
  H: number,
  cfgIn: Partial<VectorizeConfig> = {},
  imageSize?: [number, number],
): Geometry {
  const cfg = { ...DEFAULT_CONFIG, ...cfgIn };
  const label = cleanMask(labelIn, W, H, 5, cfg.minSpeckle);
  const N = W * H;
  const struct = new Uint8Array(N);
  for (let i = 0; i < N; i++) struct[i] = label[i] === WALL || label[i] === DOOR || label[i] === WINDOW ? 1 : 0;

  // ---- thickness
  let t = 4;
  const structClosed0 = struct;
  let dist = distanceTransform(structClosed0, W, H);
  let skel = skeletonize(structClosed0, W, H);
  {
    const vals: number[] = [];
    for (let i = 0; i < N; i++) if (skel[i]) vals.push(dist[i]);
    if (vals.length) t = Math.min(Math.max(2 * median(vals), 2), Math.max(4, Math.min(W, H) / 8));
  }

  // ---- walls
  const walls: Wall[] = [];
  let segs: Seg[] = [];
  if (vals0(struct)) {
    const k = Math.max(1, Math.round(t / 3));
    const m = morphClose(struct, W, H, k);
    if (k > 1) {
      dist = distanceTransform(m, W, H);
      skel = skeletonize(m, W, H);
    }
    const paths = pruneSpurs(skeletonPaths(skel, W, H), cfg.spurLen * t);
    for (const p of paths) {
      if (p.length < 2) continue;
      const pts: Pt[] = p.map((i) => [i % W, (i / W) | 0]);
      const approx = rdp(pts, Math.max(1, cfg.rdpWall * t));
      for (let i = 0; i + 1 < approx.length; i++) {
        const a = approx[i];
        const b = approx[i + 1];
        const n = Math.max(2, Math.floor(Math.hypot(b[0] - a[0], b[1] - a[1])));
        const ds: number[] = [];
        for (let j = 0; j < n; j++) {
          const x = Math.round(a[0] + ((b[0] - a[0]) * j) / (n - 1));
          const y = Math.round(a[1] + ((b[1] - a[1]) * j) / (n - 1));
          ds.push(dist[Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))]);
        }
        const s: Seg = { x1: a[0], y1: a[1], x2: b[0], y2: b[1], t: 2 * median(ds) };
        segs.push(cfg.manhattan ? snapAxis(s, cfg.snapAngleDeg) : s);
      }
    }
    if (cfg.manhattan) {
      segs = mergeCollinear(segs, cfg.mergeOffset * t, cfg.mergeGap * t);
      segs = closeJunctions(segs, cfg.junctionTol * t);
      segs = mergeCollinear(segs, cfg.mergeOffset * t, 0.5);
    }
    segs = segs.filter((s) => segLen(s) >= cfg.minWallLen * t);
    segs.forEach((s, i) =>
      walls.push({ id: i, p1: [r2(s.x1), r2(s.y1)], p2: [r2(s.x2), r2(s.y2)], thickness: r2(Math.max(1, s.t)), exterior: isExterior(s, label, W, H) }),
    );
  }

  // ---- rooms
  const rooms: Room[] = [];
  {
    const bin = new Uint8Array(N);
    for (let i = 0; i < N; i++) bin[i] = label[i] === ROOM ? 1 : 0;
    const cc = cfg.splitRadius > 0 ? splitRegions(bin, W, H, cfg.splitRadius * t) : components(bin, W, H, 4);
    const first = new Int32Array(cc.count + 1).fill(-1);
    const sums = new Float64Array((cc.count + 1) * 2);
    for (let i = 0; i < N; i++) {
      const l = cc.labels[i];
      if (!l) continue;
      if (first[l] < 0) first[l] = i;
      sums[2 * l] += i % W;
      sums[2 * l + 1] += (i / W) | 0;
    }
    const comp = new Uint8Array(N);
    let rid = 0;
    for (let l = 1; l <= cc.count; l++) {
      if (cc.areas[l] < cfg.minRoomArea) continue;
      for (let i = 0; i < N; i++) comp[i] = cc.labels[i] === l ? 1 : 0;
      const contour = traceContour(comp, W, H, first[l]);
      let poly = rdp(contour, Math.max(1, cfg.rdpRoom * t), true);
      if (poly.length < 3) continue;
      if (cfg.manhattan) poly = orthogonalize(poly, cfg.snapAngleDeg);
      rooms.push({
        id: rid++,
        label: "room",
        polygon: poly.map(([x, y]) => [r2(x), r2(y)] as Pt),
        holes: [],
        area_px: Math.round(polyArea(poly) * 10) / 10,
        area_m2: null,
        centroid: [Math.round((sums[2 * l] / cc.areas[l]) * 10) / 10, Math.round((sums[2 * l + 1] / cc.areas[l]) * 10) / 10],
      });
    }
  }

  // ---- openings
  const openings: Opening[] = [];
  {
    let oid = 0;
    for (const [kind, c] of [["door", DOOR], ["window", WINDOW]] as const) {
      const bin = new Uint8Array(N);
      for (let i = 0; i < N; i++) bin[i] = label[i] === c ? 1 : 0;
      const cc = components(bin, W, H, 8);
      const members: Pt[][] = Array.from({ length: cc.count + 1 }, () => []);
      for (let i = 0; i < N; i++) if (cc.labels[i]) members[cc.labels[i]].push([i % W, (i / W) | 0]);
      for (let l = 1; l <= cc.count; l++) {
        if (cc.areas[l] < cfg.minOpeningArea) continue;
        const rr = minAreaRect(members[l]);
        let w = rr.w + 1;
        let h = rr.h + 1;
        let ang = rr.angle;
        if (h > w) {
          [w, h] = [h, w];
          ang += 90;
        }
        ang = ((((ang + 90) % 180) + 180) % 180) - 90;
        if (cfg.manhattan) {
          if (Math.abs(ang) <= cfg.snapAngleDeg) ang = 0;
          else if (Math.abs(Math.abs(ang) - 90) <= cfg.snapAngleDeg) ang = 90;
        }
        openings.push({
          id: oid++,
          kind,
          polygon: boxPoints(rr.cx, rr.cy, w, h, ang),
          center: [r2(rr.cx), r2(rr.cy)],
          width: r2(w),
          depth: r2(h),
          angle: r2(ang),
          wall_id: hostWall([rr.cx, rr.cy], walls, 2 * t + 2),
        });
      }
    }
  }

  let geo: Geometry = {
    width: W,
    height: H,
    rooms,
    walls,
    openings,
    wall_polygons: [],
    px_per_meter: cfg.pxPerMeter,
    meta: { wall_thickness_px: r2(t), taxonomy: "coarse", engine: "floorplannet-ts" },
  };
  if (imageSize && (imageSize[0] !== W || imageSize[1] !== H)) geo = rescale(geo, imageSize[0] / W, imageSize[1] / H);
  if (cfg.pxPerMeter) for (const r of geo.rooms) r.area_m2 = r2(r.area_px / cfg.pxPerMeter ** 2);
  return geo;
}

function vals0(a: Uint8Array): boolean {
  for (let i = 0; i < a.length; i++) if (a[i]) return true;
  return false;
}

function isExterior(s: Seg, label: Uint8Array, W: number, H: number): boolean {
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L;
  const ny = dx / L;
  const off = s.t * 1.5 + 2;
  let bg = 0;
  let total = 0;
  for (const f of [0.2, 0.35, 0.5, 0.65, 0.8]) {
    const cx = s.x1 + dx * f;
    const cy = s.y1 + dy * f;
    for (const sg of [1, -1]) {
      const x = Math.round(cx + sg * nx * off);
      const y = Math.round(cy + sg * ny * off);
      total++;
      if (x < 0 || y < 0 || x >= W || y >= H || label[y * W + x] === 0) bg++;
    }
  }
  return bg >= Math.max(2, Math.floor(total / 4));
}

export function hostWall(p: Pt, walls: Wall[], maxD: number): number | null {
  let best: number | null = null;
  let bd = maxD;
  for (const w of walls) {
    const d = pointSegDist(p, w.p1, w.p2);
    if (d < bd) {
      bd = d;
      best = w.id;
    }
  }
  return best;
}

export function pointSegDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  const u = L2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return Math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy));
}

export function rescale(geo: Geometry, sx: number, sy: number): Geometry {
  const P = (p: Pt): Pt => [r2(p[0] * sx), r2(p[1] * sy)];
  const s = (sx + sy) / 2;
  return {
    ...geo,
    width: Math.round(geo.width * sx),
    height: Math.round(geo.height * sy),
    rooms: geo.rooms.map((r) => ({
      ...r,
      polygon: r.polygon.map(P),
      holes: r.holes.map((h) => h.map(P)),
      centroid: P(r.centroid),
      area_px: Math.round(r.area_px * sx * sy * 10) / 10,
    })),
    walls: geo.walls.map((w) => ({ ...w, p1: P(w.p1), p2: P(w.p2), thickness: r2(w.thickness * s) })),
    openings: geo.openings.map((o) => ({ ...o, polygon: o.polygon.map(P), center: P(o.center), width: r2(o.width * s), depth: r2(o.depth * s) })),
    wall_polygons: geo.wall_polygons.map((p) => p.map(P)),
    meta: { ...geo.meta, wall_thickness_px: r2(Number(geo.meta.wall_thickness_px || 0) * s) },
  };
}

export function summary(g: Geometry) {
  return {
    rooms: g.rooms.length,
    walls: g.walls.length,
    doors: g.openings.filter((o) => o.kind === "door").length,
    windows: g.openings.filter((o) => o.kind === "window").length,
    wallLength: g.walls.reduce((a, w) => a + Math.hypot(w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]), 0),
    roomArea: g.rooms.reduce((a, r) => a + r.area_px, 0),
  };
}
