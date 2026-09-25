import { polyArea } from "./vectorize";
import type { Geometry, Opening, Pt, Room, Wall } from "./types";

export type Selection = { type: "room" | "wall" | "opening"; id: number } | null;

const r2 = (v: number) => Math.round(v * 100) / 100;

function centroid(p: Pt[]): Pt {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i];
    const [x2, y2] = p[(i + 1) % p.length];
    const c = x1 * y2 - x2 * y1;
    a += c;
    cx += (x1 + x2) * c;
    cy += (y1 + y2) * c;
  }
  if (Math.abs(a) < 1e-9) return p[0] ?? [0, 0];
  return [r2(cx / (3 * a)), r2(cy / (3 * a))];
}

function withRoom(r: Room, polygon: Pt[], ppm: number | null): Room {
  const area = Math.round(polyArea(polygon) * 10) / 10;
  return { ...r, polygon, area_px: area, area_m2: ppm ? r2(area / ppm ** 2) : null, centroid: centroid(polygon) };
}

export function moveRoomVertex(g: Geometry, id: number, idx: number, p: Pt): Geometry {
  return {
    ...g,
    rooms: g.rooms.map((r) => {
      if (r.id !== id) return r;
      const poly = r.polygon.slice();
      poly[idx] = [r2(p[0]), r2(p[1])];
      return withRoom(r, poly, g.px_per_meter);
    }),
  };
}

export function insertRoomVertex(g: Geometry, id: number, p: Pt): Geometry {
  return {
    ...g,
    rooms: g.rooms.map((r) => {
      if (r.id !== id) return r;
      let best = 0;
      let bd = Infinity;
      for (let i = 0; i < r.polygon.length; i++) {
        const a = r.polygon[i];
        const b = r.polygon[(i + 1) % r.polygon.length];
        const d = segDist(p, a, b);
        if (d < bd) {
          bd = d;
          best = i;
        }
      }
      const poly = r.polygon.slice();
      poly.splice(best + 1, 0, [r2(p[0]), r2(p[1])]);
      return withRoom(r, poly, g.px_per_meter);
    }),
  };
}

export function removeRoomVertex(g: Geometry, id: number, idx: number): Geometry {
  return {
    ...g,
    rooms: g.rooms.map((r) => (r.id !== id || r.polygon.length <= 3 ? r : withRoom(r, r.polygon.filter((_, i) => i !== idx), g.px_per_meter))),
  };
}

export function moveWallEnd(g: Geometry, id: number, end: 1 | 2, p: Pt): Geometry {
  return { ...g, walls: g.walls.map((w) => (w.id !== id ? w : { ...w, [end === 1 ? "p1" : "p2"]: [r2(p[0]), r2(p[1])] })) };
}

export function translate(g: Geometry, sel: NonNullable<Selection>, dx: number, dy: number): Geometry {
  const T = (q: Pt): Pt => [r2(q[0] + dx), r2(q[1] + dy)];
  if (sel.type === "opening")
    return { ...g, openings: g.openings.map((o) => (o.id !== sel.id ? o : { ...o, polygon: o.polygon.map(T), center: T(o.center) })) };
  if (sel.type === "wall") return { ...g, walls: g.walls.map((w) => (w.id !== sel.id ? w : { ...w, p1: T(w.p1), p2: T(w.p2) })) };
  return { ...g, rooms: g.rooms.map((r) => (r.id !== sel.id ? r : { ...r, polygon: r.polygon.map(T), centroid: T(r.centroid) })) };
}

export function addWall(g: Geometry, p1: Pt, p2: Pt, thickness: number): Geometry {
  const id = Math.max(-1, ...g.walls.map((w) => w.id)) + 1;
  const w: Wall = { id, p1: [r2(p1[0]), r2(p1[1])], p2: [r2(p2[0]), r2(p2[1])], thickness: r2(thickness), exterior: false };
  return { ...g, walls: [...g.walls, w] };
}

export function addRoomRect(g: Geometry, a: Pt, b: Pt): Geometry {
  const id = Math.max(-1, ...g.rooms.map((r) => r.id)) + 1;
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
  const room: Room = { id, label: "room", polygon: [], holes: [], area_px: 0, area_m2: null, centroid: [0, 0] };
  return { ...g, rooms: [...g.rooms, withRoom(room, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => [r2(x), r2(y)] as Pt), g.px_per_meter)] };
}

export function addOpening(g: Geometry, kind: Opening["kind"], center: Pt, width: number, depth: number, angle: number): Geometry {
  const id = Math.max(-1, ...g.openings.map((o) => o.id)) + 1;
  const t = (angle * Math.PI) / 180;
  const ux = Math.cos(t), uy = Math.sin(t), vx = -uy, vy = ux;
  const hw = width / 2, hh = depth / 2;
  const [cx, cy] = center;
  const polygon: Pt[] = [
    [cx - ux * hw - vx * hh, cy - uy * hw - vy * hh],
    [cx + ux * hw - vx * hh, cy + uy * hw - vy * hh],
    [cx + ux * hw + vx * hh, cy + uy * hw + vy * hh],
    [cx - ux * hw + vx * hh, cy - uy * hw + vy * hh],
  ].map(([x, y]) => [r2(x), r2(y)] as Pt);
  return { ...g, openings: [...g.openings, { id, kind, polygon, center, width: r2(width), depth: r2(depth), angle, wall_id: null }] };
}

export function remove(g: Geometry, sel: NonNullable<Selection>): Geometry {
  if (sel.type === "room") return { ...g, rooms: g.rooms.filter((r) => r.id !== sel.id) };
  if (sel.type === "wall")
    return { ...g, walls: g.walls.filter((w) => w.id !== sel.id), openings: g.openings.map((o) => (o.wall_id === sel.id ? { ...o, wall_id: null } : o)) };
  return { ...g, openings: g.openings.filter((o) => o.id !== sel.id) };
}

export function updateRoom(g: Geometry, id: number, patch: Partial<Room>): Geometry {
  return { ...g, rooms: g.rooms.map((r) => (r.id === id ? { ...r, ...patch } : r)) };
}

export function updateWall(g: Geometry, id: number, patch: Partial<Wall>): Geometry {
  return { ...g, walls: g.walls.map((w) => (w.id === id ? { ...w, ...patch } : w)) };
}

export function updateOpening(g: Geometry, id: number, patch: Partial<Opening>): Geometry {
  return { ...g, openings: g.openings.map((o) => (o.id === id ? { ...o, ...patch } : o)) };
}

export function setScale(g: Geometry, ppm: number | null): Geometry {
  return { ...g, px_per_meter: ppm, rooms: g.rooms.map((r) => ({ ...r, area_m2: ppm ? r2(r.area_px / ppm ** 2) : null })) };
}

/** Axis-snap ``p`` to the x / y of reference points within ``tol``. */
export function snapTo(p: Pt, refs: Pt[], tol: number): Pt {
  let [x, y] = p;
  let bx = tol, by = tol;
  for (const r of refs) {
    const dx = Math.abs(r[0] - p[0]);
    const dy = Math.abs(r[1] - p[1]);
    if (dx < bx) { bx = dx; x = r[0]; }
    if (dy < by) { by = dy; y = r[1]; }
  }
  return [x, y];
}

function segDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  const u = L2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return Math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy));
}
