/* 🗂️ Polls the Task Center overview for the /tasks page and the nav badge.
   Same visibility discipline as useTrainingActivity: polling PAUSES while the
   tab is hidden and resumes with an immediate refresh. */
import { useEffect, useState } from 'react'
import { apiFetch } from '../api/fetchClient'
import { EMPTY_OVERVIEW, normalizeOverview } from '../utils/taskOverview'

const PAGE_POLL_MS = 5000

export function useTaskOverview(enabled = true) {
  const [overview, setOverview] = useState(EMPTY_OVERVIEW)

  useEffect(() => {
    if (!enabled) return
    let alive = true
    let timer = null

    const tick = async () => {
      try {
        const data = await apiFetch('/api/tasks/overview', { background: true })
        if (alive) setOverview(normalizeOverview(data))
      } catch {
        // Keep the last-known state on a transient error; clearing it would
        // blink the badge off, which reads as "nothing is running".
      }
    }

    const schedule = () => {
      clearInterval(timer)
      if (!document.hidden) timer = setInterval(tick, PAGE_POLL_MS)
    }

    const onVisibility = () => {
      if (!document.hidden) tick()
      schedule()
    }

    tick()
    schedule()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      alive = false
      clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [enabled])

  return overview
}
