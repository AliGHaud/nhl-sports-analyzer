import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Uncomment to proxy API calls during local dev, if needed:
    // proxy: { '/api': 'http://localhost:8000' },
  },
})
