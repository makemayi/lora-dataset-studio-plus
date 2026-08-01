import {
  DOCKER_UPDATE_COMMANDS,
  DOCKER_UPDATE_GUIDE_URL,
} from '../settings/updateStatus'

/** Shared Docker update guidance. A container's /app belongs to the image, so
 * replacing it in place is both unreliable and temporary. The Settings card
 * and global update banner intentionally render this exact same component. */
export default function DockerUpdateInstructions() {
  return (
    <div className="w-full space-y-2 rounded-md border border-sky-400/30 bg-sky-500/10 px-3 py-2 text-content">
      <p className="text-sm">
        This Docker install is updated from the host checkout. In the repository folder, run:
      </p>
      <ol className="space-y-1" aria-label="Docker update commands">
        {DOCKER_UPDATE_COMMANDS.map((command, index) => (
          <li key={command} className="flex min-w-0 items-start gap-2">
            <span aria-hidden className="w-4 shrink-0 text-right text-xs text-content-subtle">{index + 1}.</span>
            <code className="min-w-0 overflow-x-auto whitespace-nowrap rounded bg-app/70 px-2 py-1 text-xs text-content">
              {command}
            </code>
          </li>
        ))}
      </ol>
      <p className="text-xs text-content-muted">
        Compose rebuilds and recreates the container; datasets and settings in the mounted data folder stay in place.{' '}
        <a href={DOCKER_UPDATE_GUIDE_URL} target="_blank" rel="noreferrer"
          className="font-medium text-sky-300 underline hover:text-sky-200">
          Docker GPU update guide ↗
        </a>
      </p>
    </div>
  )
}
