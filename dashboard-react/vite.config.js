import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSingleFile } from 'vite-plugin-singlefile'

// The gateway serves the built app from its own origin (:9100), so every fetch in
// src/ is a same-origin relative path like `/dashboard/data`. That is right in
// production and wrong in `npm run dev`, where the app is served by Vite on :5173
// and those paths hit Vite instead of the gateway — which answers with the SPA's
// index.html, and the caller reports `Unexpected token '<', "<!doctype "`.
//
// The proxy below forwards the API paths to the gateway during dev only. It is
// not part of the build, so production behaviour is unchanged.
const GATEWAY = process.env.GATEWAY_URL || 'http://127.0.0.1:9100'

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  server: {
    proxy: {
      // Every gateway-owned path the dashboard calls. `/dashboard` covers
      // data, logs, planner-trace, run-steps and screenshot; `/targets` covers
      // the profile list, validation and knowledge upload.
      '/dashboard': { target: GATEWAY, changeOrigin: true },
      '/targets': { target: GATEWAY, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    cssCodeSplit: false,
    assetsInlineLimit: 100000000,
    chunkSizeWarningLimit: 5000,
  },
})
