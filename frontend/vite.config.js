import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/* Where `npm run dev` sends /api.
 *
 * This used to be the literal string below, hard-coded — and 127.0.0.1:5050 is
 * where the app you actually USE is listening. So a dev server started to try a
 * UI change drove the real install: every POST, every delete, every training
 * launch landed in real data. It worked, which is exactly the problem — nothing
 * ever said which backend was being talked to.
 *
 * The default is unchanged, so the habit ("npm run dev", hit :5173) still works.
 * Point it somewhere else with LDS_DEV_API_TARGET, in the shell or in
 * frontend/.env.local:
 *
 *     LDS_DEV_API_TARGET=http://127.0.0.1:5051 npm run dev
 *
 * Prefixed LDS_ rather than VITE_ on purpose: VITE_* variables are inlined into
 * the CLIENT bundle, and this is a dev-server setting that has no business
 * shipping in built output. */
const DEFAULT_DEV_API_TARGET = 'http://127.0.0.1:5050'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), ['LDS_'])
  const target = process.env.LDS_DEV_API_TARGET || env.LDS_DEV_API_TARGET
    || DEFAULT_DEV_API_TARGET
  return {
    plugins: [react()],
    base: '/',
    build: {
      outDir: 'dist',
      emptyOutDir: true,
    },
    server: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/api': target,
      },
    },
  }
})
