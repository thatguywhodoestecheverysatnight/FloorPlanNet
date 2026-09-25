# FloorPlanNet web

Next.js app that runs the FloorPlanNet DeepLabV3+ model in the browser (onnxruntime-web, WASM) and vectorises the mask with a TypeScript port of the Python vectoriser. Results are editable (move vertices, draw walls and rooms, insert doors and windows, undo and redo) and export to SVG, DXF, GeoJSON and JSON.

```bash
npm install        # also copies the ONNX Runtime WASM files into public/ort
npm run dev        # http://localhost:3000
npm test           # vectoriser unit + parity tests
npm run build
```

Deploy on Vercel with the project root set to `web`. See `.env.example` for optional settings (`NEXT_PUBLIC_API_URL` switches the default engine source for the Server API mode).

Keyboard: `V` select, `H` pan, `W` wall, `R` room, `D` door, `N` window, `Del` delete, `Ctrl+Z` undo, `Ctrl+Shift+Z` redo, scroll to zoom.
