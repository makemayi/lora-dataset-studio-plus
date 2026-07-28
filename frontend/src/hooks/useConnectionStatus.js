/* 📡 React view onto the shared connection state (utils/connectionStatus.js).
   Kept as a plain .js hook so the store stays testable under `node --test`,
   which cannot parse JSX. */
import { useEffect, useState } from 'react'
import { getConnectionState, subscribeConnection } from '../utils/connectionStatus'

export function useConnectionStatus() {
  const [status, setStatus] = useState(getConnectionState)
  useEffect(() => {
    // Re-read on subscribe: a request may have failed between the initial
    // render and this effect.
    setStatus(getConnectionState())
    return subscribeConnection(setStatus)
  }, [])
  return status
}
