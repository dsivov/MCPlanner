/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          base: '#0a0a0c',
          panel: '#111114',
          elevated: '#18181c',
          border: '#23232a',
        },
        fg: {
          DEFAULT: '#e6e6ea',
          muted: '#9b9ba6',
          dim: '#6b6b78',
        },
        accent: {
          DEFAULT: '#7c5cff',  // violet
          alt: '#22d3ee',      // cyan
          ok: '#34d399',
          warn: '#fbbf24',
          err: '#f87171',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 24px -8px rgba(124,92,255,0.45)',
        soft: '0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -8px rgba(0,0,0,0.6)',
      },
      animation: {
        'pulse-slow': 'pulse 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'dash': 'dash 20s linear infinite',
      },
      keyframes: {
        dash: {
          to: { strokeDashoffset: '-200' },
        },
      },
    },
  },
  plugins: [],
}
