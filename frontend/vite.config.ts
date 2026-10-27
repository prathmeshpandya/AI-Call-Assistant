import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /ws and /api to the FastAPI backend, so the browser talks to one origin.
const backend = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
      "/api": { target: backend },
    },
  },
});
