// Entry for the single-folder static build (scripts/build-standalone.mjs):
// the same React app without a Next.js server, for any static host.
import { createRoot } from "react-dom/client";
import Page from "../app/page";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(<Page />);
