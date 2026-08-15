import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build is embedded in the Rust binary, so it must be relative-path safe
// and self-contained: no CDN, no absolute asset roots.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Studio is served from localhost, so one request more or less costs
    // nothing; keeping sourcemaps makes a crash report actionable.
    sourcemap: true,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5273,
    proxy: { "/api": "http://127.0.0.1:4570" },
  },
});
