import Workspace from "@/components/Workspace";
import ModelCard from "@/components/ModelCard";

const REPO = process.env.NEXT_PUBLIC_REPO_URL || "https://github.com/";

const STEPS = [
  { t: "Raster plan", d: "PNG, JPG or a scan. Resized so the long side matches the working resolution, padded to a stride-32 multiple." },
  { t: "DeepLabV3+", d: "Atrous spatial pyramid pooling plus a stride-4 decoder labels every pixel as background, wall, room, door or window." },
  { t: "Mask cleanup", d: "Speckles below a size threshold are removed and refilled from the nearest confident label." },
  { t: "Walls", d: "Wall and opening pixels are thinned to a skeleton, traced into a graph, simplified with Douglas-Peucker, snapped to 90 degrees, merged and joined at T and L junctions." },
  { t: "Rooms", d: "Each connected room region is traced into a contour and rectified so near-axis edges become exact right angles." },
  { t: "Openings", d: "Door and window blobs become oriented rectangles attached to their host wall." },
];

export default function Page() {
  return (
    <main>
      <header className="top">
        <a href="#" className="brand" aria-label="FloorPlanNet home">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
            <rect x="3" y="3" width="26" height="26" rx="3" fill="none" stroke="currentColor" strokeWidth="3" />
            <path d="M3 17h13V3M16 17v12M22 17h7" stroke="currentColor" strokeWidth="3" fill="none" />
          </svg>
          FloorPlanNet
        </a>
        <nav>
          <a href="#app">Workspace</a>
          <a href="#how">How it works</a>
          <a href="#model">Model</a>
          <a href="#deploy">API and deploy</a>
          <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </header>

      <section className="hero">
        <h1>Turn floor plan images into editable geometry.</h1>
        <p>
          FloorPlanNet segments walls, rooms, doors and windows with a DeepLabV3+ network, then vectorises the mask into clean polygons and wall
          centrelines. Edit the result, calibrate the scale, and export to SVG, DXF or GeoJSON. The default model runs entirely in your browser.
        </p>
      </section>

      <Workspace />

      <section className="doc" id="how">
        <h2>How it works</h2>
        <ol className="steps">
          {STEPS.map((s, i) => (
            <li key={s.t}>
              <span className="n">{i + 1}</span>
              <div>
                <strong>{s.t}</strong>
                <p>{s.d}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="muted">
          Every tolerance in the vectoriser is expressed as a multiple of the median wall thickness, so the same settings work on a 512 px thumbnail
          and a 4000 px scan. The TypeScript vectoriser in this page is a line-for-line port of the Python reference in the repository.
        </p>
      </section>

      <section className="doc" id="model">
        <h2>Model</h2>
        <ModelCard />
        <p className="muted">
          The repository ships benchmark configs for CubiCasa5K (5,000 annotated plans; 4,200 train, 400 val, 400 test) with DeepLabV3 ResNet-101 and
          DeepLabV3+ ResNet-50. Train them on a GPU, export to ONNX, and serve through the API below for full-resolution results.
        </p>
      </section>

      <section className="doc" id="deploy">
        <h2>API and deploy</h2>
        <div className="card-grid">
          <div className="card">
            <h4>Run the API</h4>
            <pre>{`docker compose up --build api
curl -F file=@plan.png localhost:8000/api/predict`}</pre>
            <p className="muted small">
              Returns geometry JSON, a class-index mask and timings. POST the geometry to <code>/api/export/dxf</code> (or svg, geojson) to get a
              file back.
            </p>
          </div>
          <div className="card">
            <h4>Connect this page</h4>
            <p className="small">
              Switch Inference to <b>Server API</b> and paste the URL, or set <code>NEXT_PUBLIC_API_URL</code> in Vercel so it is the default.
            </p>
            <pre>{`# vercel project settings
Root directory: web
NEXT_PUBLIC_API_URL=https://api.yourdomain.com`}</pre>
          </div>
          <div className="card">
            <h4>Train your own</h4>
            <pre>{`make data      # CubiCasa5K
make train     # DeepLabV3 R101
make eval      # test split, TTA
make export    # ONNX`}</pre>
          </div>
        </div>
      </section>

      <footer className="foot">
        <span>FloorPlanNet. MIT licensed code. CubiCasa5K data is CC BY-NC 4.0.</span>
        <span className="muted">Images you open here never leave your device unless you choose the server API.</span>
      </footer>
    </main>
  );
}
