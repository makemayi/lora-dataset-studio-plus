/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        /* The soft-pastel accent pair pulled from the reference UI: a muted
           mauve-pink brand (#d48ec2 deepened to #c983b7 for enough contrast
           on a light ground) and a sky blue (#5fb9dc) as its gradient second
           stop / secondary accent. Peach (#fad6be) is the warm third voice.
           All three live on a light grey-blue ground — Morandi-soft, never
           saturated. */
        primary: {
          DEFAULT: '#c983b7',
          dark: '#5fb9dc',
        },
        accent: {
          DEFAULT: '#5fb9dc',
          peach: '#fad6be',
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
        'gradient-primary': 'linear-gradient(135deg, rgba(255 255 255 / 0.10) 0%, rgba(255 255 255 / 0) 42%), linear-gradient(135deg, #c983b7 0%, #5fb9dc 100%)',
      },
    },
  },
  plugins: [],
}
