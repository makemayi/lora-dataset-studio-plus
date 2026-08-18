/* Nav badge for the Task Center: amber count = in-progress tasks, red dot =
   failures the user has not seen yet. Seen failures are remembered in
   localStorage so the red dot survives page reloads until the Tasks page
   is opened. */
import { useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router'
import { useTaskOverview } from '../../hooks/useTaskOverview'
import { badgeCounts } from '../../utils/taskOverview'

const SEEN_KEY = 'taskFailuresSeen'

function readSeen() {
  try { return new Set(JSON.parse(localStorage.getItem(SEEN_KEY) || '[]')) }
  catch { return new Set() }
}

export default function TaskNavBadge() {
  const overview = useTaskOverview(true)
  const location = useLocation()
  const [seen, setSeen] = useState(readSeen)

  // Opening the Tasks page marks every currently-visible failure as seen.
  useEffect(() => {
    if (location.pathname !== '/tasks') return
    const merged = new Set([...seen, ...overview.failedIds])
    setSeen(merged)
    try { localStorage.setItem(SEEN_KEY, JSON.stringify([...merged])) } catch { /* ignore */ }
  }, [location.pathname, overview.failedIds]) // eslint-disable-line react-hooks/exhaustive-deps

  const { running, failed } = useMemo(
    () => badgeCounts(overview, seen), [overview, seen])
  if (running === 0 && failed === 0) return null
  return (
    <span className="ml-1 inline-flex items-center gap-1">
      {running > 0 && (
        <span role="status" title={`${running} task${running === 1 ? '' : 's'} in progress`}
              className="rounded-full bg-amber-50 px-1.5 text-[0.625rem] font-semibold text-amber-700">
          {running}
        </span>
      )}
      {failed > 0 && (
        <span role="status" title={`${failed} failed task${failed === 1 ? '' : 's'}`}
              className="h-2 w-2 rounded-full bg-red-500" />
      )}
    </span>
  )
}
