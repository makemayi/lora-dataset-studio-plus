/** A synchronous guard for async bulk requests. React state updates on the next
 * render; this gate changes immediately, so a second click in the same frame
 * cannot launch a second destructive request. */
export function createBulkActionGate() {
  let active = null
  return {
    begin(action, count) {
      if (active) return null
      active = { action, count }
      return active
    },
    finish(token) {
      if (active === token) active = null
    },
    get active() { return active },
  }
}

export function bulkActionMessage(active) {
  if (!active) return ''
  if (active.action === 'delete') {
    return `Deleting ${active.count} ${active.count === 1 ? 'image' : 'images'}…`
  }
  return `Updating ${active.count} images…`
}
