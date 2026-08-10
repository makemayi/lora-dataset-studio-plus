/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        /* The accent, softened on 2026-08-10 with the rest of the tone pass.
           #5567d5 → #6a74cc keeps the same indigo but drops a third of its
           chroma: on a neutral graphite ground the old one read as the loudest
           thing on screen, above the photographs. `dark` is the second stop of
           gradient-primary and moved the same way — from a saturated violet
           (#764ba2) to one a shade off the accent, so the gradient reads as
           depth rather than as two different colours. */
        primary: {
          DEFAULT: '#6a74cc',
          dark: '#7b6bb4',
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
        'gradient-primary': 'linear-gradient(135deg, #6069bd 0%, #7264a8 100%)',
      },
    },
  },
  plugins: [],
}
