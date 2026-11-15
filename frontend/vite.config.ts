import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

declare const process: { env: Record<string, string | undefined> };

// `npm run dev` serves the app on :5173 and forwards /api to the FastAPI server on :8000.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 }, // one ~230 kB (gzip) bundle: React + Recharts
  server: { proxy: { "/api": process.env.API_URL ?? "http://localhost:8000" } },
});
