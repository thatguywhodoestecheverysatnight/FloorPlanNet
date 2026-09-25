"use client";
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Editor, { Layers, Tool } from "./Editor";
import * as E from "@/lib/edit";
import { download, toDXF, toGeoJSON, toSVG } from "@/lib/exporters";
import { segment } from "@/lib/model";
import { asset } from "@/lib/config";
import { DEFAULT_API, apiHealth, predictRemote } from "@/lib/api";
import { summary, vectorize } from "@/lib/vectorize";
import { CLASS_COLORS, CLASSES, Geometry, ROOM_TYPES, Segmentation } from "@/lib/types";

const SAMPLES = [1, 2, 3, 4, 5, 6].map((i) => ({ src: asset(`samples/plan-${i}.png`), name: `Sample ${i}` }));

type Status = { kind: "idle" | "busy" | "done" | "error"; msg: string };
type Hist = { past: Geometry[]; present: Geometry | null; committed: Geometry | null; future: Geometry[] };

const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: "select", label: "Select", key: "V", hint: "Click to select, drag to move, drag handles to reshape. Double click a room edge to add a vertex, Alt+click a vertex to remove it." },
  { id: "pan", label: "Pan", key: "H", hint: "Drag to pan. Scroll to zoom anywhere." },
  { id: "wall", label: "Wall", key: "W", hint: "Drag to draw a wall. Snaps to 90 degrees and nearby vertices (hold Shift for free angle)." },
  { id: "room", label: "Room", key: "R", hint: "Drag a rectangle to add a room." },
  { id: "door", label: "Door", key: "D", hint: "Click on a wall to insert a door." },
  { id: "window", label: "Window", key: "N", hint: "Click on a wall to insert a window." },
];

function maskToUrl(seg: Segmentation): string {
  const c = document.createElement("canvas");
  c.width = seg.width;
  c.height = seg.height;
  const ctx = c.getContext("2d")!;
  const im = ctx.createImageData(seg.width, seg.height);
  for (let i = 0; i < seg.label.length; i++) {
    const k = seg.label[i];
    const [r, g, b] = CLASS_COLORS[k] || [0, 0, 0];
    im.data[4 * i] = r;
    im.data[4 * i + 1] = g;
    im.data[4 * i + 2] = b;
    im.data[4 * i + 3] = k === 0 ? 0 : 255;
  }
  ctx.putImageData(im, 0, 0);
  return c.toDataURL("image/png");
}

function maskPng(seg: Segmentation): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = seg.width;
  c.height = seg.height;
  const ctx = c.getContext("2d")!;
  const im = ctx.createImageData(seg.width, seg.height);
  for (let i = 0; i < seg.label.length; i++) {
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = seg.label[i];
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  return new Promise((res) => c.toBlob((b) => res(b!), "image/png"));
}

export default function Workspace() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState("floorplan");
  const [size, setSize] = useState({ w: 0, h: 0 });
  const imgRef = useRef<HTMLImageElement | null>(null);
  const blobRef = useRef<Blob | null>(null);
  const [seg, setSeg] = useState<Segmentation | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [hist, setHist] = useState<Hist>({ past: [], present: null, committed: null, future: [] });
  const [status, setStatus] = useState<Status>({ kind: "idle", msg: "Load a floor plan to begin." });
  const [timing, setTiming] = useState<{ seg: number; vec: number } | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [selection, setSelection] = useState<E.Selection>(null);
  const [layers, setLayers] = useState<Layers>({ image: true, mask: false, rooms: true, walls: true, openings: true, labels: true, imageOpacity: 0.85 });
  const [engine, setEngine] = useState<"browser" | "api">("browser");
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [maxSide, setMaxSide] = useState(512);
  const [tta, setTta] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [calib, setCalib] = useState("");
  const g = hist.present;

  const commit = useCallback((next: Geometry) => {
    setHist((h) => ({ past: h.committed ? [...h.past.slice(-99), h.committed] : h.past, present: next, committed: next, future: [] }));
  }, []);
  const preview = useCallback((next: Geometry) => setHist((h) => ({ ...h, present: next })), []);
  const undo = useCallback(
    () =>
      setHist((h) => {
        if (!h.past.length || !h.committed) return h;
        const prev = h.past[h.past.length - 1];
        return { past: h.past.slice(0, -1), present: prev, committed: prev, future: [h.committed, ...h.future] };
      }),
    [],
  );
  const redo = useCallback(
    () =>
      setHist((h) => {
        if (!h.future.length || !h.committed) return h;
        const [nx, ...rest] = h.future;
        return { past: [...h.past, h.committed], present: nx, committed: nx, future: rest };
      }),
    [],
  );

  const run = useCallback(
    async (img: HTMLImageElement) => {
      setSelection(null);
      try {
        setStatus({ kind: "busy", msg: engine === "browser" ? "Loading model" : "Calling API" });
        let s: Segmentation;
        let geo: Geometry;
        let vecMs = 0;
        if (engine === "api") {
          if (!apiUrl) throw new Error("Set the API URL first (see the Deploy section).");
          const blob = blobRef.current ?? (await (await fetch(img.src)).blob());
          const r = await predictRemote(apiUrl, blob, { maxSide: Math.max(maxSide, 1024), tta });
          s = r.seg;
          geo = r.geometry;
        } else {
          s = await segment(img, maxSide, (msg) => setStatus({ kind: "busy", msg }));
          setStatus({ kind: "busy", msg: "Vectorising with contour analysis" });
          await new Promise((r) => setTimeout(r, 16));
          const t0 = performance.now();
          geo = vectorize(s.label, s.width, s.height, {}, [img.naturalWidth, img.naturalHeight]);
          vecMs = performance.now() - t0;
        }
        setSeg(s);
        setMaskUrl(maskToUrl(s));
        setHist({ past: [], present: geo, committed: geo, future: [] });
        setTiming({ seg: s.ms, vec: vecMs });
        const sm = summary(geo);
        setStatus({ kind: "done", msg: `Found ${sm.rooms} rooms, ${sm.walls} walls, ${sm.doors} doors, ${sm.windows} windows.` });
      } catch (err) {
        console.error(err);
        setStatus({ kind: "error", msg: err instanceof Error ? err.message : String(err) });
      }
    },
    [engine, apiUrl, maxSide, tta],
  );

  const loadImage = useCallback(
    (src: string, name: string, blob: Blob | null, autorun = true) => {
      const img = new Image();
      img.onload = () => {
        imgRef.current = img;
        blobRef.current = blob;
        setImageUrl(src);
        setImageName(name.replace(/\.[^.]+$/, "") || "floorplan");
        setSize({ w: img.naturalWidth, h: img.naturalHeight });
        setSeg(null);
        setMaskUrl(null);
        setHist({ past: [], present: null, committed: null, future: [] });
        if (autorun) void run(img);
      };
      img.onerror = () => setStatus({ kind: "error", msg: "Could not read that image. Use PNG, JPG or WebP." });
      img.src = src;
    },
    [run],
  );

  const onFile = (f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setStatus({ kind: "error", msg: "Please choose an image file (PNG, JPG, WebP). For PDFs, export a page as PNG first." });
      return;
    }
    loadImage(URL.createObjectURL(f), f.name, f);
  };

  // open the first sample on first visit so the pipeline is visible immediately
  const booted = useRef(false);
  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    loadImage(SAMPLES[0].src, "sample-1", null, true);
  }, [loadImage]);

  useEffect(() => {
    if (engine !== "api" || !apiUrl) return setApiOk(null);
    let alive = true;
    apiHealth(apiUrl).then((ok) => alive && setApiOk(ok));
    return () => {
      alive = false;
    };
  }, [engine, apiUrl]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        return e.shiftKey ? redo() : undo();
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        return redo();
      }
      if ((e.key === "Delete" || e.key === "Backspace") && selection && g) {
        e.preventDefault();
        commit(E.remove(g, selection));
        setSelection(null);
        return;
      }
      if (e.key === "Escape") return setSelection(null);
      const t = TOOLS.find((x) => x.key.toLowerCase() === e.key.toLowerCase());
      if (t && !mod) setTool(t.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selection, g, commit, undo, redo]);

  const stats = useMemo(() => (g ? summary(g) : null), [g]);
  const ppm = g?.px_per_meter ?? null;
  const fmtLen = (px: number) => (ppm ? `${(px / ppm).toFixed(2)} m` : `${Math.round(px)} px`);
  const fmtArea = (px: number) => (ppm ? `${(px / ppm ** 2).toFixed(1)} m²` : `${Math.round(px).toLocaleString()} px²`);
  const selRoom = selection?.type === "room" ? g?.rooms.find((r) => r.id === selection.id) : undefined;
  const selWall = selection?.type === "wall" ? g?.walls.find((w) => w.id === selection.id) : undefined;
  const selOpen = selection?.type === "opening" ? g?.openings.find((o) => o.id === selection.id) : undefined;
  const classShare = useMemo(() => {
    if (!seg) return null;
    const c = new Array(CLASSES.length).fill(0);
    for (let i = 0; i < seg.label.length; i++) c[seg.label[i]]++;
    return c.map((v) => v / seg.label.length);
  }, [seg]);

  const exportAs = async (fmt: "svg" | "geojson" | "json" | "dxf" | "mask") => {
    if (!g) return;
    const base = `${imageName}_floorplannet`;
    if (fmt === "svg") download(`${base}.svg`, toSVG(g), "image/svg+xml");
    if (fmt === "geojson") download(`${base}.geojson`, JSON.stringify(toGeoJSON(g), null, 1), "application/geo+json");
    if (fmt === "json") download(`${base}.json`, JSON.stringify(g, null, 1), "application/json");
    if (fmt === "dxf") download(`${base}.dxf`, toDXF(g), "application/dxf");
    if (fmt === "mask" && seg) download(`${base}_mask.png`, await maskPng(seg));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    onFile(e.dataTransfer.files?.[0]);
  };

  const importJson = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const geo = JSON.parse(await f.text()) as Geometry;
      if (!Array.isArray(geo.rooms) || !Array.isArray(geo.walls)) throw new Error("not a FloorPlanNet geometry file");
      setSize({ w: geo.width, h: geo.height });
      commit(geo);
      setStatus({ kind: "done", msg: `Imported ${f.name}` });
    } catch (err) {
      setStatus({ kind: "error", msg: `Import failed: ${err instanceof Error ? err.message : err}` });
    }
    e.target.value = "";
  };

  return (
    <section className="workspace" id="app">
      <aside className="panel left">
        <div
          className={`drop ${dragOver ? "over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <label className="dropinner">
            <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} hidden />
            <strong>Drop a floor plan</strong>
            <span>or click to browse (PNG, JPG, WebP)</span>
          </label>
        </div>
        <h4>Samples</h4>
        <div className="samples">
          {SAMPLES.map((s) => (
            <button key={s.src} className="sample" onClick={() => loadImage(s.src, s.name, null)} title={s.name}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={s.src} alt={s.name} loading="lazy" />
            </button>
          ))}
        </div>

        <h4>Inference</h4>
        <div className="seg">
          <button className={engine === "browser" ? "on" : ""} onClick={() => setEngine("browser")}>In browser</button>
          <button className={engine === "api" ? "on" : ""} onClick={() => setEngine("api")}>Server API</button>
        </div>
        {engine === "browser" ? (
          <label className="field">
            Working resolution
            <select value={maxSide} onChange={(e) => setMaxSide(Number(e.target.value))}>
              <option value={384}>384 px (fastest)</option>
              <option value={512}>512 px (default)</option>
              <option value={768}>768 px</option>
              <option value={1024}>1024 px (large plans)</option>
            </select>
          </label>
        ) : (
          <>
            <label className="field">
              API URL
              <input value={apiUrl} placeholder="https://your-api.example.com" onChange={(e) => setApiUrl(e.target.value.trim())} />
            </label>
            <div className="apistate">
              {apiOk === null ? "Not checked" : apiOk ? <span className="ok">Connected</span> : <span className="bad">Unreachable</span>}
            </div>
            <label className="check">
              <input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} /> Test-time augmentation
            </label>
          </>
        )}
        <button className="primary" disabled={!imgRef.current || status.kind === "busy"} onClick={() => imgRef.current && run(imgRef.current)}>
          {status.kind === "busy" ? "Working..." : "Run FloorPlanNet"}
        </button>
        <p className={`status ${status.kind}`} role="status" aria-live="polite">
          {status.kind === "busy" && <span className="spinner" />} {status.msg}
        </p>
        {timing && (
          <p className="muted small">
            Segmentation {Math.round(timing.seg)} ms{timing.vec ? `, vectorisation ${Math.round(timing.vec)} ms` : ""}
            {seg ? `, mask ${seg.width}x${seg.height}, confidence ${(seg.meanConfidence * 100).toFixed(1)}%` : ""}
          </p>
        )}
      </aside>

      <div className="stage">
        <div className="toolbar" role="toolbar" aria-label="Editing tools">
          {TOOLS.map((t) => (
            <button key={t.id} className={tool === t.id ? "on" : ""} onClick={() => setTool(t.id)} title={`${t.label} (${t.key})`} disabled={!g && t.id !== "pan" && t.id !== "select"}>
              {t.label} <kbd>{t.key}</kbd>
            </button>
          ))}
          <span className="sep" />
          <button onClick={undo} disabled={!hist.past.length} title="Undo (Ctrl+Z)">Undo</button>
          <button onClick={redo} disabled={!hist.future.length} title="Redo (Ctrl+Shift+Z)">Redo</button>
          <button
            onClick={() => {
              if (g && selection) {
                commit(E.remove(g, selection));
                setSelection(null);
              }
            }}
            disabled={!selection}
            title="Delete selection (Del)"
          >
            Delete
          </button>
        </div>
        <p className="hint">{TOOLS.find((t) => t.id === tool)?.hint}</p>
        {imageUrl || g ? (
          <Editor
            imageUrl={imageUrl}
            maskUrl={maskUrl}
            geometry={g}
            width={size.w}
            height={size.h}
            layers={layers}
            tool={tool}
            selection={selection}
            onSelect={setSelection}
            onPreview={preview}
            onCommit={commit}
          />
        ) : (
          <div className="empty">
            <div>
              <h3>No plan loaded</h3>
              <p>Pick a sample on the left or drop your own raster floor plan. Everything runs locally in your browser; nothing is uploaded.</p>
            </div>
          </div>
        )}
      </div>

      <aside className="panel right">
        <h4>Layers</h4>
        <div className="layers">
          {(["image", "mask", "rooms", "walls", "openings", "labels"] as const).map((k) => (
            <label key={k} className="check">
              <input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} />
              {k === "mask" ? "segmentation mask" : k}
            </label>
          ))}
          <label className="field">
            Image opacity
            <input type="range" min={0} max={1} step={0.05} value={layers.imageOpacity} onChange={(e) => setLayers({ ...layers, imageOpacity: Number(e.target.value) })} />
          </label>
        </div>
        {classShare && (
          <div className="legend">
            {CLASSES.map((c, i) => (
              <div key={c} className="legrow">
                <span className="sw" style={{ background: `rgb(${CLASS_COLORS[i].join(",")})` }} />
                {c}
                <span className="muted">{(classShare[i] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}

        {stats && (
          <>
            <h4>Plan</h4>
            <dl className="stats">
              <div><dt>Rooms</dt><dd>{stats.rooms}</dd></div>
              <div><dt>Walls</dt><dd>{stats.walls}</dd></div>
              <div><dt>Doors</dt><dd>{stats.doors}</dd></div>
              <div><dt>Windows</dt><dd>{stats.windows}</dd></div>
              <div><dt>Wall length</dt><dd>{fmtLen(stats.wallLength)}</dd></div>
              <div><dt>Room area</dt><dd>{fmtArea(stats.roomArea)}</dd></div>
            </dl>
            {!ppm && <p className="muted small">Select a wall and enter its real length to calibrate metres.</p>}
          </>
        )}

        {(selRoom || selWall || selOpen) && g && (
          <>
            <h4>Selection</h4>
            <div className="inspector">
              {selRoom && (
                <>
                  <label className="field">
                    Room type
                    <input list="roomtypes" value={selRoom.label} onChange={(e) => commit(E.updateRoom(g, selRoom.id, { label: e.target.value }))} />
                    <datalist id="roomtypes">{ROOM_TYPES.map((r) => <option key={r} value={r} />)}</datalist>
                  </label>
                  <p className="muted small">{selRoom.polygon.length} vertices, {fmtArea(selRoom.area_px)}</p>
                </>
              )}
              {selWall && (
                <>
                  <p className="small">
                    {selWall.exterior ? "Exterior" : "Interior"} wall, {fmtLen(Math.hypot(selWall.p2[0] - selWall.p1[0], selWall.p2[1] - selWall.p1[1]))}
                  </p>
                  <label className="field">
                    Thickness (px)
                    <input type="number" min={1} step={0.5} value={selWall.thickness} onChange={(e) => commit(E.updateWall(g, selWall.id, { thickness: Math.max(1, Number(e.target.value)) }))} />
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={selWall.exterior} onChange={(e) => commit(E.updateWall(g, selWall.id, { exterior: e.target.checked }))} /> exterior
                  </label>
                  <div className="row">
                    <input className="grow" type="number" min={0} step={0.01} placeholder="real length (m)" value={calib} onChange={(e) => setCalib(e.target.value)} />
                    <button
                      onClick={() => {
                        const m = Number(calib);
                        if (!(m > 0)) return;
                        const L = Math.hypot(selWall.p2[0] - selWall.p1[0], selWall.p2[1] - selWall.p1[1]);
                        commit(E.setScale(g, L / m));
                        setCalib("");
                      }}
                    >
                      Calibrate
                    </button>
                  </div>
                </>
              )}
              {selOpen && (
                <>
                  <label className="field">
                    Kind
                    <select value={selOpen.kind} onChange={(e) => commit(E.updateOpening(g, selOpen.id, { kind: e.target.value as "door" | "window" }))}>
                      <option value="door">door</option>
                      <option value="window">window</option>
                    </select>
                  </label>
                  <p className="muted small">width {fmtLen(selOpen.width)}, host wall {selOpen.wall_id ?? "none"}</p>
                </>
              )}
            </div>
          </>
        )}

        <h4>Export</h4>
        <div className="exports">
          <button disabled={!g} onClick={() => exportAs("svg")}>SVG</button>
          <button disabled={!g} onClick={() => exportAs("dxf")}>DXF (CAD)</button>
          <button disabled={!g} onClick={() => exportAs("geojson")}>GeoJSON</button>
          <button disabled={!g} onClick={() => exportAs("json")}>JSON</button>
          <button disabled={!seg} onClick={() => exportAs("mask")}>Mask PNG</button>
          <label className="button">
            Import JSON
            <input type="file" accept=".json,application/json" onChange={importJson} hidden />
          </label>
        </div>
        {ppm && <p className="muted small">Scale: {ppm.toFixed(2)} px per metre</p>}

        {g && g.rooms.length > 0 && (
          <>
            <h4>Rooms</h4>
            <ul className="list">
              {g.rooms.map((r) => (
                <li key={r.id}>
                  <button className={selection?.type === "room" && selection.id === r.id ? "on" : ""} onClick={() => setSelection({ type: "room", id: r.id })}>
                    <span>#{r.id} {r.label}</span>
                    <span className="muted">{fmtArea(r.area_px)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </aside>
    </section>
  );
}
