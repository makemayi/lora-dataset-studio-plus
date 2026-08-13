/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        /* The accent, brought BACK to life on 2026-08-13. The 2026-08-10 pass
           had desaturated it to #6470b4 so it would not compete with the
           photographs; in practice it faded into the graphite until the UI
           read as grey-on-grey. #6b74e8 is a confident indigo that still sits
           clearly apart from the semantic green (kept) and the amber/sky
           engine accents, and `dark` (#8b5cf6) is the violet second stop of
           gradient-primary — one colour family, two depths. */
        primary: {
          DEFAULT: '#0e6b54',
          dark: '#0d9488',
        },
        accent: {
          DEFAULT: '#f97316',
          peach: '#fdba74',
        },
        // ── Semantic theme tokens (backed by CSS vars in index.css) ──────────
        // App is dark-only. The *-alpha-baked tokens (surface, surface-raised,
        // border, border-strong) carry CSS-var-controlled default opacity.
        // Use the *-solid variants when you need to set your own alpha via
        // Tailwind's /NN modifier.
        app: 'rgb(var(--bg-app) / <alpha-value>)',
        surface: 'rgb(var(--surface) / var(--surface-alpha))',
        'surface-raised': 'rgb(var(--surface-raised) / var(--surface-raised-alpha))',
        'surface-overlay': 'rgb(var(--surface-overlay) / <alpha-value>)',
        'surface-solid': 'rgb(var(--surface-overlay) / <alpha-value>)',
        content: 'rgb(var(--content) / <alpha-value>)',
        'content-muted': 'rgb(var(--content-muted) / <alpha-value>)',
        'content-subtle': 'rgb(var(--content-subtle) / <alpha-value>)',
        border: 'rgb(var(--border) / var(--border-alpha))',
        'border-strong': 'rgb(var(--border-strong) / var(--border-strong-alpha))',
      },
      backgroundImage: {
        /* A real indigo→violet sweep again: the flattened version read as grey
           paint. A white top sheen is layered ahead of it so buttons read as
           having material. The sweep stays one colour family so it announces
           the action without announcing a second colour. */
        'gradient-primary': 'linear-gradient(135deg, rgba(255 255 255 / 0.10) 0%, rgba(255 255 255 / 0) 42%), linear-gradient(135deg, #0e6b54 0%, #0d9488 100%)',
      },
    },
  },
  plugins: [],
}
