/* 🏋️ Polls "is anything training right now" for the nav indicator.
   Separate from useCapabilities on purpose: capabilities are probed and cached
   server-side for 30 s, which is the wrong freshness for a live indicator, and
   probing every 15 s would be wasteful. GET /api/train/activity is one flag
   plus one COUNT.

   Polling PAUSES while the tab is hidden and resumes with an immediate refresh
   — this app is consulted from a phone, and a background tab quietly waking a
   request every 15 s is exactly how a page earns its battery reputation. */
import { useEffect, useState } from 'react'
import { apiFetch } from '../api/fetchClient'
import { EMPTY_ACTIVITY, normalizeActivity } from '../utils/trainingActivity'

const POLL_MS = 15000

export function useTrainingActivity() {
  const [activity, setActivity] = useState(EMPTY_ACTIVITY)

  useEffect(() => {
    let alive = true
    let timer = null

    const tick = async () => {
      try {
        const data = await apiFetch('/api/train/activity', { background: true })
        if (alive) setActivity(normalizeActivity(data))
      } catch {
        // Keep the last-known state on a transient error. Clearing it would
        // blink the indicator off mid-training, which reads as "it stopped".
      }
    }

    const schedule = () => {
      clearInterval(timer)
      if (!document.hidden) timer = setInterval(tick, POLL_MS)
    }

    const onVisibility = () => {
      if (!document.hidden) tick()   // don't make the user wait a full period
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
  }, [])

  return activity
}
