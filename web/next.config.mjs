/** @type {import('next').NextConfig} */
const isolation = [
  // Cross-origin isolation unlocks SharedArrayBuffer, so onnxruntime-web can use WASM threads.
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [
      { source: "/:path*", headers: [...isolation, { key: "X-Content-Type-Options", value: "nosniff" }] },
      { source: "/models/:file*", headers: [{ key: "Cache-Control", value: "public, max-age=604800, immutable" }] },
      { source: "/ort/:file*", headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }] },
    ];
  },
};

export default nextConfig;
