// Shared by ReferencePanel and PoseSlotPanel: hover a tile, Ctrl+V an image
// instead of always going through the file picker.
export function imageFromClipboard(e) {
  const items = e.clipboardData?.items;
  if (!items) return null;
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) return item.getAsFile();
  }
  return null;
}
