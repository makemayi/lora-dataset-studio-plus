/* The composed prompt — what the engine ACTUALLY receives.
   ------------------------------------------------------------------------
   WHY THIS PANEL EXISTS
   A local-edit prompt is ~1000 characters assembled from six sources: the
   command, the shot description (with the outfit/expression directives baked in),
   the per-framing detail, the markings hold, the identity lock and the rendering
   tail. Every one of them is now editable — and until this panel, none of them
   was visible. Reading one meant writing a throwaway script; editing one meant
   changing a sixth of a string you had never seen whole.

   It sits directly under the fields, in the same card, because that is the only
   place it is useful: the point is to edit a part and watch the whole move.

   COMPOSED SERVER-SIDE, ON PURPOSE. The text comes from POST
   /api/settings/prompt-preview, which calls the same wrap_variation_* functions
   generation calls. Re-implementing the assembly in JS would have produced a
   preview that drifts from reality on the first backend change — and a preview
   you cannot trust is worse than none. It is a pure text call: no model, no GPU,
   nothing enqueued, nothing billed.

   UNSAVED TEXT. Settings saves on an explicit button, so the request carries the
   in-flight `identity_prompts` tree: the preview shows what you are typing, not
   what you last saved. */
import { useEffect, useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { HelpText } from './primitives'

const ENGINES = [
  { id: 'krea', label: 'Krea (local)' },
  { id: 'klein', label: 'Klein (local)' },
  { id: 'nanobanana', label: 'Nano Banana' },
  { id: 'chatgpt', label: 'ChatGPT' },
  { id: 'openrouter', label: 'OpenRouter' },
]
const FRAMINGS = [
  { id: 'face', label: 'Face' },
  { id: 'bust', label: 'Bust' },
  { id: 'body', label: 'Full body' },
  { id: 'back', label: 'Back' },
]

const SELECT_CLASS =
  'w-full rounded-md border border-border-strong bg-surface-raised px-2 py-1.5 text-xs text-content ' +
  'focus:border-primary focus:outline-none'

/** The API engines never see a framing detail, a rendering tail or a markings
 *  hold — wrap_variation is guard-first and has no such parts. Saying so beats
 *  letting the user wonder why their tail edit does not show up in the preview. */
const API_ENGINES = ['nanobanana', 'chatgpt', 'openrouter']

export default function PromptPreview({ subject, identityPrompts }) {
  const [engine, setEngine] = useState('krea')
  const [framing, setFraming] = useState('bust')
  const [nsfw, setNsfw] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  const body = JSON.stringify(identityPrompts || {})
  useEffect(() => {
    let alive = true
    // Debounced: this fires on every keystroke in every prompt box above, and the
    // panel is a text composition, not something to hammer 60 times a second.
    const t = setTimeout(() => {
      postJson('/api/settings/prompt-preview', {
        engine, framing, nsfw, subject_type: subject, identity_prompts: JSON.parse(body),
      })
        .then((d) => { if (alive) { setData(d); setError(null) } })
        // A failed preview must never look like a failed SETTING: it is a
        // read-only aid, so it reports itself and leaves the fields alone.
        .catch((e) => { if (alive) setError(e?.message || 'Preview unavailable') })
    }, 250)
    return () => { alive = false; clearTimeout(t) }
  }, [engine, framing, nsfw, subject, body])

  const isApi = API_ENGINES.includes(engine)
  const text = data?.prompt || ''

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard blocked (http origin / permissions) — the text is selectable */ }
  }

  return (
    <div id="prompt-preview" className="border-t border-border pt-4">
      <h4 className="text-sm font-medium text-content">What actually gets sent</h4>
      <HelpText className="mt-1 mb-2 text-xs text-content-muted">
        The full prompt for one real shot, assembled from every box on this card, as the
        engine receives it. Type above and watch it change — nothing is generated, nothing
        is saved and nothing is billed by this panel.
      </HelpText>

      {/* Two columns from 400px (the two selects), the toggle on its own row. A
          three-across row of selects is what overflows a phone card. */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label htmlFor="prompt-preview-engine" className="block text-[0.6875rem] font-medium text-content-muted">
            Engine
          </label>
          <select
            id="prompt-preview-engine"
            value={engine}
            onChange={(e) => setEngine(e.target.value)}
            className={SELECT_CLASS}
          >
            {ENGINES.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="prompt-preview-framing" className="block text-[0.6875rem] font-medium text-content-muted">
            Shot
          </label>
          <select
            id="prompt-preview-framing"
            value={framing}
            onChange={(e) => setFraming(e.target.value)}
            disabled={isApi}
            className={`${SELECT_CLASS} disabled:opacity-50`}
          >
            {FRAMINGS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
          </select>
        </div>
      </div>

      <label htmlFor="prompt-preview-nsfw" className="mt-2 flex items-center gap-2 text-xs text-content">
        <input
          id="prompt-preview-nsfw"
          type="checkbox"
          checked={nsfw}
          onChange={(e) => setNsfw(e.target.checked)}
          disabled={isApi}
          className="h-4 w-4 rounded border-border-strong disabled:opacity-50"
        />
        Uncensored shot (local engines only)
      </label>

      {isApi && (
        <p className="mt-2 text-xs text-content-subtle">
          This engine gets the identity lock followed by the shot description — no framing
          block, no rendering tail and no markings hold. Those four boxes only reach Klein
          and Krea.
        </p>
      )}

      {error ? (
        <p className="mt-2 rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-xs text-content-muted">
          Preview unavailable ({error}). Your prompt boxes are unaffected — this panel only reads.
        </p>
      ) : (
        <>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <span className="text-[0.6875rem] text-content-subtle">
              {data
                ? <>Shot <strong className="text-content-muted">{data.shot_label || data.shot_id}</strong> · {data.length} characters</>
                : 'Composing…'}
            </span>
            <button
              type="button"
              onClick={copy}
              disabled={!text}
              className="rounded-md border border-border-strong px-2 py-1 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          {/* THE 400px CASE: 1000 characters of prose. `whitespace-pre-wrap` +
              `break-words` keep every line inside the card (the page must never
              scroll sideways), and the box scrolls VERTICALLY at a bounded
              height so the preview never pushes the fields off the screen.
              Taller on a wide screen, where there is room for it. */}
          <pre
            aria-live="polite"
            className="mt-1 max-h-48 overflow-y-auto overflow-x-hidden rounded-md border border-border-strong
                       bg-surface-raised px-3 py-2 font-mono text-[0.6875rem] leading-relaxed text-content
                       whitespace-pre-wrap break-words sm:max-h-72 sm:text-xs"
          >
            {text || ' '}
          </pre>
        </>
      )}
    </div>
  )
}
