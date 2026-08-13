/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        /* The functional accents pulled from the reference UI: the nav's
           primary button is a vivid blue (#0b71e8), and the secondary voice is
           an indigo-purple (#6478d6). Pink/peach are DECORATIVE (illustration
           fills), not action colours — kept here as `accent.rose`/`accent.peach`
           so decorations can reach for them without hardcoding. */
        primary: {
          DEFAULT: '#4F46E5',
          dark: '#7C3AED',
        },
        accent: {
          DEFAULT: '#7C3AED',
          peach: '#fad6be',
          rose: '#d48ec2',
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
        'gradient-primary': 'linear-gradient(135deg, rgba(255 255 255 / 0.10) 0%, rgba(255 255 255 / 0) 42%), linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)',
      },
    },
  },
  plugins: [],
}
