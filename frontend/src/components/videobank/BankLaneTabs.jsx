import { NavLink } from 'react-router'
import { ImageIcon, FilmIcon } from '../common/icons'

/** The kind of bank you are making — 🖼 images or 🎬 video.
 *
 * WHY A SWITCH AND NOT A SIXTH NAV ITEM. The desktop bar already carries five
 * workspaces and overflows at exactly 768 px (see App.jsx). More to the point,
 * these are not two workspaces: they are one job — "triage a folder before it
 * becomes a dataset" — over two kinds of material. Someone with a folder of
 * rushes looks for the Bank, and today the Bank ignores every .mp4 in it
 * WITHOUT SAYING SO. This switch is where that silence ends.
 *
 * They stay two separate routes, and two separate sets of components, because
 * the work genuinely differs: a video bank cuts one file into hundreds of shots,
 * stores bounds rather than files, and encodes only at promotion.
 */
export default function BankLaneTabs({ className = '' }) {
  /* A filled pill for the active lane, nothing at all for the other one — the
     same grammar as the nav bar. The outline on both was a second separator
     doing the job the fill already does. */
  const cls = ({ isActive }) => [
    'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold no-underline transition-colors',
    isActive
      ? 'bg-surface-raised text-content'
      : 'text-content-muted hover:bg-surface hover:text-content',
  ].join(' ')
  return (
    <div role="group" aria-label="Kind of bank"
      className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      <span className="mr-1 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
        bank of
      </span>
      {/* `end` on /bank so it does not stay lit while /video-bank is open. */}
      <NavLink to="/bank" end className={cls}>
        <ImageIcon className="h-3.5 w-3.5 shrink-0" /> Images
      </NavLink>
      <NavLink to="/video-bank" className={cls}>
        <FilmIcon className="h-3.5 w-3.5 shrink-0" /> Video
        {/* The lane ships, works end to end, and is younger than everything
            around it — the badge says "expect rough edges", not "expect loss":
            triage data is the only thing at stake and sources are never written. */}
        <span className="rounded-full bg-amber-50 px-1.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
          Beta
        </span>
      </NavLink>
    </div>
  )
}
