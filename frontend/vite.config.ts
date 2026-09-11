/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        // Override for a local dev backend on a different port (e.g. a throwaway
        // instance during frontend-only testing) without touching the default every
        // other `npm run dev` still relies on.
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  // The same proxy for `vite preview`, which serves the PRODUCTION build. Vite keeps the
  // two configs separate, so without this a built bundle cannot reach the API at all.
  //
  // It exists because a performance number taken against `npm run dev` is a number about
  // the debug build: measured on hero-74 during a goto, 267.8 ms of 529.9 ms of
  // application JavaScript was `jsxDEV` - React's development-only JSX factory, which
  // carries debug source info and is stripped from a production bundle entirely. Any
  // "how long does our JavaScript take" question has to be asked of the build that ships.
  preview: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
  },
})
