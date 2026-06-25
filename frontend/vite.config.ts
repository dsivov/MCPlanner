import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Avatar tab loads `three` and `talkinghead` at runtime from a CDN, resolved by the
// importmap in index.html. They must NOT be bundled — mark them external so both the dev
// server and the production build leave the dynamic import() as a bare specifier for the
// browser's importmap to resolve.
const CDN_EXTERNAL = ['three', 'talkinghead', /^three\/addons\//]

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['three', 'talkinghead'] },
  build: {
    rollupOptions: { external: CDN_EXTERNAL },
    // rolldown-vite reads this key:
    rolldownOptions: { external: CDN_EXTERNAL },
  } as any,
})
