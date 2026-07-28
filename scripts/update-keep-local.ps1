<#
  update-keep-local.ps1 — pull upstream commits into a checkout that has its own
  local commits, without the in-app "Update & restart" button's fast-forward-only
  limitation.

  WHY THIS EXISTS
  ----------------
  The in-app updater runs `git pull --ff-only`, which by design can NEVER succeed
  once this checkout has even one local commit — that is not a bug it hits, it is
  what --ff-only means. A checkout with local fixes/features on `main` will always
  see "Couldn't fast-forward" from that button. This script does the merge the
  button can't: fetch, merge upstream in, and — since almost every real conflict
  on this repo is in the built `frontend/dist/**` (a generated artifact, never a
  source of truth) — resolve pure-dist conflicts automatically by rebuilding, the
  same way a human resolves them by hand.

  SAFETY
  ------
  - Refuses to run with uncommitted changes to tracked files (commit or stash
    first) — a merge on a dirty tree is how work gets silently lost.
  - Auto-resolves ONLY if every conflicted path is under frontend/dist/. Any
    conflict touching a real source file stops the script with the merge left
    OPEN (unresolved, not aborted) and the file list printed, so nothing is
    guessed at on your behalf. Run `git merge --abort` to back out if you want a
    clean slate before untangling it a different way.
  - Never touches `config.json`, `.env`, or anything under `data/` — those are
    gitignored and this script only ever runs `git fetch` / `git merge`.

  USAGE:  powershell -ExecutionPolicy Bypass -File scripts\update-keep-local.ps1
  Then restart the app yourself — this script does not do that part.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

# --- guard: no uncommitted changes to TRACKED files -------------------------
$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
  Write-Host "Uncommitted changes to tracked files:" -ForegroundColor Yellow
  Write-Host $dirty
  Fail "Commit or stash these first — a merge on a dirty tree can lose work. Aborting, nothing touched."
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host "Branch: $branch"

Write-Host "Fetching origin/$branch..."
git fetch --quiet origin $branch
if (-not $?) { Fail "git fetch failed — offline, or no access to the remote." }

$behind = [int](git rev-list --count "HEAD..origin/$branch").Trim()
if ($behind -eq 0) {
  Write-Host "Already up to date with origin/$branch — nothing to merge." -ForegroundColor Green
  exit 0
}
Write-Host "$behind commit(s) behind origin/$branch. Merging..."

git merge "origin/$branch" --no-edit 2>&1 | Out-Host
$mergeExit = $LASTEXITCODE

if ($mergeExit -eq 0) {
  Write-Host "Merged cleanly. Restart the app to pick it up." -ForegroundColor Green
  exit 0
}

# --- merge produced conflicts: is EVERY one of them a dist/ build artifact? --
$conflicted = (git diff --name-only --diff-filter=U) -split "`n" | Where-Object { $_ }
if (-not $conflicted) {
  Fail "git merge failed but reported no conflicted files — inspect manually with `git status`."
}

$nonDist = $conflicted | Where-Object { $_ -notlike 'frontend/dist/*' }
if ($nonDist) {
  Write-Host "Merge conflicts outside frontend/dist/ — these need a real decision, not a rebuild:" -ForegroundColor Yellow
  $nonDist | ForEach-Object { Write-Host "  $_" }
  if ($conflicted.Count -gt $nonDist.Count) {
    Write-Host "(frontend/dist/ conflicts were left alone too, since they'll regenerate once the rest is resolved.)"
  }
  Write-Host ""
  Write-Host "The merge is still OPEN. Resolve these files and 'git commit', or 'git merge --abort' to back out."
  exit 1
}

Write-Host "All conflicts are build artifacts under frontend/dist/ — rebuilding instead of hand-merging them."
foreach ($f in $conflicted) {
  git rm -f --quiet --ignore-unmatch -- $f | Out-Null
}
# `vite build` regenerates dist/index.html + dist/assets/* from source on every
# run — it never reads the old dist/ as a template — so removing the conflicted
# copies above is enough; nothing needs seeding before the build below.

Push-Location frontend
try {
  Write-Host "Running npm run build..."
  npm run build
  if (-not $?) { Pop-Location; Fail "npm run build failed — merge left OPEN, fix the build error and re-run this script." }
} finally {
  Pop-Location
}

git add frontend/dist
git commit --quiet -m "build(frontend): rebuild dist after merging origin/$branch"
if (-not $?) { Fail "git commit failed after the dist rebuild — check `git status`." }

Write-Host "Merged and rebuilt dist. Restart the app to pick it up." -ForegroundColor Green
