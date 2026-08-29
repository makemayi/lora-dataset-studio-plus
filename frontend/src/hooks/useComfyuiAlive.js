import { useEffect, useState } from 'react'
import { apiFetch } from '../api/fetchClient'

/**
 * Is ComfyUI up? The CHEAP question, asked on its own.
 *
 * WHY THIS EXISTS
 * Every ComfyUI-gated surface used to read `caps.studio_visible` off
 * /api/capabilities. That endpoint also lists models and asks /object_info —
 * MEASURED at 37 s cold on a real install, cached 30 s afterwards, and it starts
 * from an all-false placeholder. So on a machine where ComfyUI was running the
 * whole time, the Studio page said "Test Studio needs ComfyUI" and the menu item
 * greyed out. `/api/system/comfyui-alive` costs a TCP connect plus a 37-byte GET
 * (≈5 ms here) and is cached for seconds on the server.
 *
 * Returns `null` while the answer is not known yet — callers MUST treat that as
 * "not a no". Claiming ComfyUI is missing before having asked is the whole bug.
 */
export function useComfyuiAlive(intervalMs = 15000) {
  const [alive, setAlive] = useState(null)

  useEffect(() => {
    let stopped = false
    let timer = null
    const tick = async () => {
      try {
        const d = await apiFetch('/api/system/comfyui-alive')
        if (!stopped) setAlive(Boolean(d && d.alive))
      } catch {
        // A failed poll is not a verdict: keep the last-known answer rather than
        // turning a hiccup in our OWN server into "your ComfyUI is gone".
      }
      if (!stopped) timer = setTimeout(tick, intervalMs)
    }
    tick()
    return () => { stopped = true; clearTimeout(timer) }
  }, [intervalMs])

  return alive
}

export default useComfyuiAlive
