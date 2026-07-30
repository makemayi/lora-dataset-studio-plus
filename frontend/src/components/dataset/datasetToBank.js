/* Dataset → Bank is deliberately a copy, but it now has a second decision:
   whether the copied rows should retain captions and analysis that stays valid
   for the copied image, or begin as a new unanalysed bank. Keep the request
   shape here (outside JSX) so the frontend/backend contract is easy to test. */

export const datasetToBankUrl = () => '/api/bank/from-dataset';

export function datasetToBankRequest(datasetId, name, preserveAnalysis = true) {
  return {
    dataset_id: Number(datasetId),
    name: String(name || '').trim(),
    // The API defaults this to true for older callers. Sending it explicitly
    // makes the user’s "Start fresh" choice durable and unambiguous.
    preserve_analysis: preserveAnalysis !== false,
  };
}

export function canStartDatasetToBank({ name, busy = false } = {}) {
  return !busy && Boolean(String(name || '').trim());
}
