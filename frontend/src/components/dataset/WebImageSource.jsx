import { webImageSource } from '../../utils/webImageSource';

export default function WebImageSource({ metadata, className = '' }) {
  const origin = webImageSource(metadata);
  if (!origin) return null;
  return (
    <a href={origin.sourceUrl} target="_blank" rel="noopener noreferrer"
      className={`underline decoration-white/20 underline-offset-2 hover:text-content ${className}`}
      title={`Found on ${origin.host}`}>
      Source ↗
    </a>
  );
}
