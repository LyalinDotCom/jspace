import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API = `http://127.0.0.1:${process.env.API_PORT || 8731}`

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT) || 5173,
    proxy: {
      '/ws': { target: API, ws: true },
      '/status': API,
      '/reload_jlens': API,
    },
  },
})
