"use client";
import { PointerEvent as RPointerEvent, WheelEvent as RWheelEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as E from "@/lib/edit";
import { pointSegDist } from "@/lib/vectorize";
import type { Geometry, Pt } from "@/lib/types";

export type Tool = "select" | "pan" | "wall" | "room" | "door" | "window";

export interface Layers {
  image: boolean;
  mask: boolean;
  rooms: boolean;
  walls: boolean;
  openings: boolean;
  labels: boolean;
  imageOpacity: number;
}

interface Props {
  imageUrl: string | null;
  maskUrl: string | null;
  geometry: Geometry | null;
  width: number;
  height: number;
  layers: Layers;
  tool: Tool;
  selection: E.Selection;
  onSelect: (s: E.Selection) => void;
  /** live update (no history entry) */
  onPreview: (g: Geometry) => void;
  /** committed update (history entry) */
  onCommit: (g: Geometry) => void;
}

type Drag =
  | { kind: "pan"; sx: number; sy: number; vx: number; vy: number }
  | { kind: "vertex"; id: number; idx: number; base: Geometry }
  | { kind: "wallEnd"; id: number; end: 1 | 2; base: Geometry }
  | { kind: "move"; sel: NonNullable<E.Selection>; start: Pt; base: Geometry }
  | { kind: "draw"; tool: "wall" | "room"; start: Pt; cur: Pt };

export default function Editor(p: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [view, setView] = useState({ x: 0, y: 0, w: p.width || 1000, h: p.height || 800 });
  const [drag, setDrag] = useState<Drag | null>(null);
  const [shift, setShift] = useState(false);

  useEffect(() => {
    if (p.width && p.height) setView({ x: 0, y: 0, w: p.width, h: p.height });
  }, [p.width, p.height]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => setShift(e.shiftKey);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", down);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", down);
    };
  }, []);

  // user units per screen pixel (preserveAspectRatio=meet -> the larger ratio wins)
  const [client, setClient] = useState({ w: 900, h: 700 });
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setClient({ w: el.clientWidth || 900, h: el.clientHeight || 700 }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const unit = Math.max(view.w / client.w, view.h / client.h);
  const g = p.geometry;
  const medianT = useMemo(() => {
    if (!g?.walls.length) return Math.max(4, (p.width || 500) / 120);
    const t = g.walls.map((w) => w.thickness).sort((a, b) => a - b);
    return t[t.length >> 1];
  }, [g, p.width]);

  const toSvg = useCallback((e: { clientX: number; clientY: number }): Pt => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = svg.getScreenCTM()!.inverse();
    const q = pt.matrixTransform(m);
    return [q.x, q.y];
  }, []);

  const onWheel = (e: RWheelEvent) => {
    const [mx, my] = toSvg(e);
    const f = Math.exp(e.deltaY * 0.0015);
    setView((v) => {
      const w = Math.min(Math.max(v.w * f, 40), (p.width || 1000) * 4);
      const h = (w * v.h) / v.w;
      return { x: mx - ((mx - v.x) * w) / v.w, y: my - ((my - v.y) * h) / v.h, w, h };
    });
  };

  const axisSnap = (a: Pt, b: Pt): Pt => {
    if (shift) return b;
    return Math.abs(b[0] - a[0]) > Math.abs(b[1] - a[1]) ? [b[0], a[1]] : [a[0], b[1]];
  };

  const allVertices = useMemo<Pt[]>(() => {
    if (!g) return [];
    return [...g.rooms.flatMap((r) => r.polygon), ...g.walls.flatMap((w) => [w.p1, w.p2])];
  }, [g]);

  const placeOpening = (kind: "door" | "window", at: Pt) => {
    if (!g) return;
    let best = null as null | (typeof g.walls)[number];
    let bd = Infinity;
    for (const w of g.walls) {
      const d = pointSegDist(at, w.p1, w.p2);
      if (d < bd) {
        bd = d;
        best = w;
      }
    }
    if (!best || bd > best.thickness * 3 + 6) return;
    const dx = best.p2[0] - best.p1[0], dy = best.p2[1] - best.p1[1];
    const L2 = dx * dx + dy * dy || 1;
    const u = Math.max(0, Math.min(1, ((at[0] - best.p1[0]) * dx + (at[1] - best.p1[1]) * dy) / L2));
    const c: Pt = [best.p1[0] + u * dx, best.p1[1] + u * dy];
    const ang = (Math.atan2(dy, dx) * 180) / Math.PI;
    const next = E.addOpening(g, kind, c, best.thickness * (kind === "door" ? 3.4 : 4.5), best.thickness + 1, ang);
    next.openings[next.openings.length - 1].wall_id = best.id;
    p.onCommit(next);
    p.onSelect({ type: "opening", id: next.openings[next.openings.length - 1].id });
  };

  const onBgDown = (e: RPointerEvent) => {
    const pt = toSvg(e);
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (p.tool === "pan" || e.button === 1) {
      setDrag({ kind: "pan", sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y });
      return;
    }
    if (!g) return;
    if (p.tool === "wall" || p.tool === "room") {
      const s = E.snapTo(pt, allVertices, 8 * unit);
      setDrag({ kind: "draw", tool: p.tool, start: s, cur: s });
      return;
    }
    if (p.tool === "door" || p.tool === "window") {
      placeOpening(p.tool, pt);
      return;
    }
    p.onSelect(null);
  };

  const startItem = (e: RPointerEvent, sel: NonNullable<E.Selection>) => {
    if (p.tool !== "select" || !g) return;
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    p.onSelect(sel);
    setDrag({ kind: "move", sel, start: toSvg(e), base: g });
  };

  const onMove = (e: RPointerEvent) => {
    if (!drag) return;
    if (drag.kind === "pan") {
      const k = unit;
      setView((v) => ({ ...v, x: drag.vx - (e.clientX - drag.sx) * k, y: drag.vy - (e.clientY - drag.sy) * k }));
      return;
    }
    const pt = toSvg(e);
    if (drag.kind === "draw") {
      setDrag({ ...drag, cur: drag.tool === "wall" ? axisSnap(drag.start, E.snapTo(pt, allVertices, 8 * unit)) : pt });
      return;
    }
    if (drag.kind === "vertex") {
      const room = drag.base.rooms.find((r) => r.id === drag.id)!;
      const n = room.polygon.length;
      const refs = shift ? [] : [room.polygon[(drag.idx + n - 1) % n], room.polygon[(drag.idx + 1) % n], ...allVertices];
      p.onPreview(E.moveRoomVertex(drag.base, drag.id, drag.idx, E.snapTo(pt, refs, 6 * unit)));
    } else if (drag.kind === "wallEnd") {
      const w = drag.base.walls.find((q) => q.id === drag.id)!;
      const other = drag.end === 1 ? w.p2 : w.p1;
      const q = shift ? pt : axisSnap(other, E.snapTo(pt, allVertices, 6 * unit));
      p.onPreview(E.moveWallEnd(drag.base, drag.id, drag.end, q));
    } else if (drag.kind === "move") {
      p.onPreview(E.translate(drag.base, drag.sel, pt[0] - drag.start[0], pt[1] - drag.start[1]));
    }
  };

  const onUp = () => {
    if (!drag) return;
    if (drag.kind === "draw" && g) {
      const d = Math.hypot(drag.cur[0] - drag.start[0], drag.cur[1] - drag.start[1]);
      if (d > 4 * unit) {
        const next = drag.tool === "wall" ? E.addWall(g, drag.start, drag.cur, medianT) : E.addRoomRect(g, drag.start, drag.cur);
        p.onCommit(next);
        p.onSelect(drag.tool === "wall" ? { type: "wall", id: next.walls[next.walls.length - 1].id } : { type: "room", id: next.rooms[next.rooms.length - 1].id });
      }
    } else if (drag.kind !== "pan" && drag.kind !== "draw" && g) {
      if (g !== drag.base) p.onCommit(g);
    }
    setDrag(null);
  };

  const sel = p.selection;
  const selRoom = sel?.type === "room" ? g?.rooms.find((r) => r.id === sel.id) : undefined;
  const selWall = sel?.type === "wall" ? g?.walls.find((w) => w.id === sel.id) : undefined;
  const hr = 5 * unit;
  const cursor = p.tool === "pan" ? (drag?.kind === "pan" ? "grabbing" : "grab") : p.tool === "select" ? "default" : "crosshair";

  const fit = () => setView({ x: 0, y: 0, w: p.width, h: p.height });
  const zoom = (f: number) =>
    setView((v) => {
      const w = v.w * f, h = v.h * f;
      return { x: v.x + (v.w - w) / 2, y: v.y + (v.h - h) / 2, w, h };
    });

  return (
    <div className="editor">
      <div className="zoombar">
        <button onClick={() => zoom(0.8)} aria-label="Zoom in">+</button>
        <button onClick={() => zoom(1.25)} aria-label="Zoom out">−</button>
        <button onClick={fit}>Fit</button>
      </div>
      <svg
        ref={svgRef}
        className="canvas"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ cursor }}
        onWheel={onWheel}
        onPointerDown={onBgDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        role="img"
        aria-label="Floor plan editor"
      >
        <rect x={-1e5} y={-1e5} width={2e5} height={2e5} className="bg" />
        <rect x={0} y={0} width={p.width} height={p.height} fill="#fff" />
        {p.imageUrl && p.layers.image && (
          <image href={p.imageUrl} x={0} y={0} width={p.width} height={p.height} opacity={p.layers.imageOpacity} style={{ pointerEvents: "none" }} />
        )}
        {p.maskUrl && p.layers.mask && (
          <image href={p.maskUrl} x={0} y={0} width={p.width} height={p.height} opacity={0.55} style={{ pointerEvents: "none", imageRendering: "pixelated" }} preserveAspectRatio="none" />
        )}
        {g && p.layers.rooms && (
          <g>
            {g.rooms.map((r) => (
              <polygon
                key={`r${r.id}`}
                points={r.polygon.map((q) => q.join(",")).join(" ")}
                className={`room ${sel?.type === "room" && sel.id === r.id ? "sel" : ""}`}
                strokeWidth={1.5 * unit}
                onPointerDown={(e) => startItem(e, { type: "room", id: r.id })}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  if (g) p.onCommit(E.insertRoomVertex(g, r.id, toSvg(e)));
                }}
              />
            ))}
          </g>
        )}
        {g && p.layers.walls && (
          <g>
            {g.walls.map((w) => (
              <line
                key={`w${w.id}`}
                x1={w.p1[0]} y1={w.p1[1]} x2={w.p2[0]} y2={w.p2[1]}
                className={`wall ${w.exterior ? "ext" : ""} ${sel?.type === "wall" && sel.id === w.id ? "sel" : ""}`}
                strokeWidth={Math.max(w.thickness, 2 * unit)}
                onPointerDown={(e) => startItem(e, { type: "wall", id: w.id })}
              />
            ))}
          </g>
        )}
        {g && p.layers.openings && (
          <g>
            {g.openings.map((o) => (
              <polygon
                key={`o${o.id}`}
                points={o.polygon.map((q) => q.join(",")).join(" ")}
                className={`opening ${o.kind} ${sel?.type === "opening" && sel.id === o.id ? "sel" : ""}`}
                strokeWidth={1.2 * unit}
                onPointerDown={(e) => startItem(e, { type: "opening", id: o.id })}
              />
            ))}
          </g>
        )}
        {g && p.layers.labels && p.layers.rooms && (
          <g className="labels" style={{ pointerEvents: "none" }}>
            {g.rooms.map((r) => (
              <g key={`l${r.id}`}>
                <text x={r.centroid[0]} y={r.centroid[1]} fontSize={13 * unit} textAnchor="middle">{r.label}</text>
                <text x={r.centroid[0]} y={r.centroid[1] + 14 * unit} fontSize={11 * unit} textAnchor="middle" className="sub">
                  {r.area_m2 != null ? `${r.area_m2.toFixed(1)} m²` : `${Math.round(r.area_px)} px²`}
                </text>
              </g>
            ))}
          </g>
        )}
        {selRoom &&
          selRoom.polygon.map((q, i) => (
            <circle
              key={`h${i}`}
              cx={q[0]} cy={q[1]} r={hr}
              className="handle"
              strokeWidth={1.5 * unit}
              onPointerDown={(e) => {
                e.stopPropagation();
                (e.target as Element).setPointerCapture?.(e.pointerId);
                if (e.altKey && g) {
                  p.onCommit(E.removeRoomVertex(g, selRoom.id, i));
                  return;
                }
                if (g) setDrag({ kind: "vertex", id: selRoom.id, idx: i, base: g });
              }}
            />
          ))}
        {selWall &&
          ([1, 2] as const).map((end) => {
            const q = end === 1 ? selWall.p1 : selWall.p2;
            return (
              <circle
                key={`we${end}`}
                cx={q[0]} cy={q[1]} r={hr}
                className="handle"
                strokeWidth={1.5 * unit}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  (e.target as Element).setPointerCapture?.(e.pointerId);
                  if (g) setDrag({ kind: "wallEnd", id: selWall.id, end, base: g });
                }}
              />
            );
          })}
        {drag?.kind === "draw" &&
          (drag.tool === "wall" ? (
            <line x1={drag.start[0]} y1={drag.start[1]} x2={drag.cur[0]} y2={drag.cur[1]} className="wall preview" strokeWidth={medianT} />
          ) : (
            <rect
              x={Math.min(drag.start[0], drag.cur[0])}
              y={Math.min(drag.start[1], drag.cur[1])}
              width={Math.abs(drag.cur[0] - drag.start[0])}
              height={Math.abs(drag.cur[1] - drag.start[1])}
              className="room preview"
              strokeWidth={1.5 * unit}
            />
          ))}
      </svg>
    </div>
  );
}
