import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En desarrollo, el API corre aparte: `uv run osky-remix serve` (127.0.0.1:8765)
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/files": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});
