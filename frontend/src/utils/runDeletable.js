/* Can a lineage run be removed from the graph? Pure predicate, no React, so the
   rule is unit-testable with node:test.

   Only a GONE run qualifies — one the graph already badges as having no
   checkpoints on disk (checkpoint_ready === false, the exact condition
   lineageChrome's SavesChip draws "gone" for). A run with checkpoints on disk
   (checkpoint_ready === true) is a recoverable run and is never offered for
   deletion; a run whose availability we couldn't determine (null/undefined — an
   active run or a scan that failed) is left alone rather than guessed removable.

   A FULL MODEL is the case that made this rule worth stating twice. Its weights
   are not addressed by the checkpoint columns at all: a run delivered to Hugging
   Face only has nothing on this disk and never did, so the old rule badged it
   "gone" and offered to remove it — for a model that exists, in a repository the
   run record holds the only pointer to. The backend now answers `null` for it
   (already refused above) AND names it in `dense_artifact`; this second clause
   is what makes the refusal explicit rather than a lucky tri-state, and what
   keeps it true if the flag ever comes back as a boolean. */
export function isRunDeletable(node) {
  if (!node || node.checkpoint_ready !== false) return false;
  return node.dense_artifact !== 'local' && node.dense_artifact !== 'hub';
}

/* Drop a run from a {nodes, edges} lineage tree WITHOUT a refetch: remove the
   node, and every edge touching it. A child that resumed from the removed run
   loses its parent edge and re-roots on its own — mirroring the backend, which
   detaches (never deletes) a living child. Returns a new tree; the input is
   untouched. Robust to a missing/empty tree. */
export function removeRunFromTree(tree, recordId) {
  const nodes = Array.isArray(tree?.nodes) ? tree.nodes : [];
  const edges = Array.isArray(tree?.edges) ? tree.edges : [];
  const gone = (id) => id === recordId;
  return {
    ...tree,
    nodes: nodes.filter((n) => !gone(n.record_id)),
    edges: edges.filter((e) => !gone(e.parent) && !gone(e.child)),
  };
}
