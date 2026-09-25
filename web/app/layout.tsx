import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FloorPlanNet | Floor plan parsing and vectorisation",
  description:
    "Segment rooms, walls, doors and windows in raster floor plans with DeepLabV3+, then vectorise them into editable geometry you can export to SVG, DXF and GeoJSON.",
  openGraph: {
    title: "FloorPlanNet",
    description: "DeepLabV3+ floor plan parsing with OpenCV-style vectorisation, running in your browser.",
    type: "website",
  },
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7f9" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1216" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
