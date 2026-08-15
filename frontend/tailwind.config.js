/** @type {import('tailwindcss').Config} */

/* The brand indigo — the ONE place it lives. Every other mention (the
   gradient, the selection glow, the focus ring) derives from this string, so
   the colour cannot drift into a second value. `rgba(...)` consumers use the
   RGB triplet computed below; anything that needs the hex reads BRAND. */
const BRAND = '#4F46E5';
const BRAND_RGB = [
  parseInt(BRAND.slice(1, 3), 16),
  parseInt(BRAND.slice(3, 5), 16),
  parseInt(BRAND.slice(5, 7), 16),
].join(',');

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
          DEFAULT: BRAND,
          dark: '#7C3AED',
        },
        accent: {
          DEFAULT: '#7C3AED',
          peach: '#fad6be',
          rose: '#d48ec2',
        },
        // ── Semantic theme tokens (backed by CSS vars in index.css) ──────────
        // App is LIGHT-only since the Vision-Pro glass pass (2026-08-13/14):
        // index.html carries data-theme="light" and index.css defines only
        // `:root, [data-theme="light"]`. `darkMode` below is vestigial — there
        // is no dark block for it to select, so a `dark:` variant never
        // matches. The *-alpha-baked tokens (surface, surface-raised,
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
        'gradient-primary': `linear-gradient(135deg, rgba(255 255 255 / 0.10) 0%, rgba(255 255 255 / 0) 42%), linear-gradient(135deg, ${BRAND} 0%, #7C3AED 100%)`,
      },
      boxShadow: {
        /* Selected dataset tile: a spread of brand indigo replacing the resting
           shadow, and its hover twin so the glow SURVIVES the hover lift
           (FLOAT_HOVER's hover shadow would replace it — selection is the
           state, not the gesture). Both derive from BRAND above. */
        'tile-selected': `0 0 0 3px rgba(${BRAND_RGB},0.55), 0 0 0 8px rgba(${BRAND_RGB},0.18), 0 8px 24px rgba(${BRAND_RGB},0.30)`,
        'tile-selected-hover': `0 0 0 3px rgba(${BRAND_RGB},0.55), 0 0 0 8px rgba(${BRAND_RGB},0.18), 0 16px 40px rgba(${BRAND_RGB},0.38)`,
      },
    },
  },
  plugins: [],
}
