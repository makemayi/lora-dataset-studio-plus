#requires -Version 5.1
<#
.SYNOPSIS
  Transactional one-click updater for the Windows Docker installation.

.DESCRIPTION
  The default STABLE channel resolves the latest published GitHub Release and
  resolves that reference to one immutable commit, then downloads the archive
  for that exact commit. MAIN is an explicit preview channel resolved the same
  way; it never needs a Release to exist.

  Before changing the installation, the script rejects unsafe ZIP paths,
  symlinks/reparse points, ambiguous archive roots and bundles missing the
  Docker sentinels. A ZIP installation is switched at top-level code boundaries
  with an on-volume backup and durable rollback journal. Git checkouts are
  inspected read-only and refused with manual fast-forward instructions; this
  updater never checks out, resets or merges a Git working tree.

  User state and Docker bind mounts are excluded by name and are never copied,
  moved, removed or permission-adjusted. The new launcher is called only after
  the code switch, with --update-rebuild, which returns 0 only once Docker
  reports the Studio container healthy. That zero commits the transaction; any
  other code restores the old code and attempts its launcher once.
#>
[CmdletBinding()]
param(
    [ValidateSet('stable', 'main')]
    [string]$Channel = 'stable',

    [string]$InstallRoot = '',

    [string]$Repository = 'perfectgf/lora-dataset-studio',

    # Test-only injection points. They are rejected unless -TestMode is
    # explicit; the public BAT launchers never pass that switch.
    [string]$ArchivePath = '',
    [string]$ArchiveUri = '',
    [string]$ReleaseMetadataPath = '',
    [string]$LauncherPath = '',
    [string]$GitRemote = '',
    [string]$TestCommit = '',
    [string]$TestFault = '',
    [string]$TestSignalPath = '',
    [int]$TestHoldMilliseconds = 0,
    [long]$TestMaxArchiveBytes = 0,
    [long]$TestMaxEntryBytes = 0,
    [long]$TestMaxExpandedBytes = 0,
    [long]$TestRequiredFreeBytes = 0,
    [switch]$TestMode,

    # Internal: the public script relaunches an exact copy from a unique temp
    # directory before it permits any in-place code switch.
    [switch]$RunningFromTemp
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:OfficialRepository = 'perfectgf/lora-dataset-studio'
$script:UserAgent = 'LoRA-Dataset-Studio-Docker-Updater/1.0 (+https://github.com/perfectgf/lora-dataset-studio)'
$script:ProtectedTopLevel = @(
    '.env',
    '.git',
    '.python',
    '.venv',
    'venv',
    'config.json',
    'data',
    'data-docker',
    'run',
    'basedir',
    'data-docker-gpu',
    'bank-images',
    'ollama-data',
    '.docker-launch-settings',
    '.docker-compose.external-comfy.override.yml',
    '.docker-gpu-settings.env',
    '.docker-gpu-settings',
    '.lds-update.lock'
)
$script:RequiredSentinels = @(
    'docker-compose.gpu.yml',
    'Dockerfile.gpu',
    'backend/run.py',
    'frontend/dist/index.html',
    'packaging/docker/studio_launch.sh'
)
$script:MaxArchiveEntries = 100000
$script:MaxArchiveBytes = 512MB
$script:MaxEntryBytes = 512MB
$script:MaxExpandedBytes = 2GB
$script:DiskReserveBytes = 256MB

function Assert-TestModeConfiguration {
    $hasInjection = [bool](
        $ArchivePath -or $ArchiveUri -or $ReleaseMetadataPath -or
        $LauncherPath -or $GitRemote -or $TestCommit -or $TestFault -or
        $TestSignalPath -or $TestHoldMilliseconds -or $TestMaxArchiveBytes -or
        $TestMaxEntryBytes -or $TestMaxExpandedBytes -or
        $TestRequiredFreeBytes -or $Repository -cne $script:OfficialRepository)
    if ($hasInjection -and -not $TestMode) {
        throw 'Injection parameters refused without explicit -TestMode.'
    }
    if ($TestFault -notin @(
            '', 'overlay-after-first-switch', 'git-dirty-before-switch',
            'hold-lock')) {
        throw "Unknown TestFault: $TestFault"
    }
    if ($TestHoldMilliseconds -lt 0 -or $TestHoldMilliseconds -gt 10000) {
        throw 'TestHoldMilliseconds must be between 0 and 10000.'
    }
    foreach ($limit in @(
            $TestMaxArchiveBytes, $TestMaxEntryBytes,
            $TestMaxExpandedBytes, $TestRequiredFreeBytes)) {
        if ($limit -lt 0) { throw 'Test limits cannot be negative.' }
    }
    if ($TestCommit -and $TestCommit -cnotmatch '^[0-9a-fA-F]{40}$') {
        throw 'TestCommit must be a full 40-character Git SHA-1.'
    }
    if ($TestFault -eq 'hold-lock' -and
            (-not $TestSignalPath -or $TestHoldMilliseconds -le 0)) {
        throw 'The hold-lock test requires TestSignalPath and a positive TestHoldMilliseconds.'
    }
    if ($TestMode) {
        if ($TestMaxArchiveBytes) { $script:MaxArchiveBytes = $TestMaxArchiveBytes }
        if ($TestMaxEntryBytes) { $script:MaxEntryBytes = $TestMaxEntryBytes }
        if ($TestMaxExpandedBytes) { $script:MaxExpandedBytes = $TestMaxExpandedBytes }
    }
}

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-ProtectedTopLevel([string]$Name) {
    foreach ($protected in $script:ProtectedTopLevel) {
        if ([string]::Equals($Name, $protected, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Get-FullPath([string]$Path) {
    return [IO.Path]::GetFullPath($Path)
}

function Assert-ChildPath([string]$Root, [string]$Candidate, [switch]$AllowRoot) {
    $rootFull = (Get-FullPath $Root).TrimEnd('\', '/')
    $candidateFull = Get-FullPath $Candidate
    if ($AllowRoot -and [string]::Equals(
            $rootFull, $candidateFull.TrimEnd('\', '/'),
            [StringComparison]::OrdinalIgnoreCase)) {
        return $candidateFull
    }
    $prefix = $rootFull + [IO.Path]::DirectorySeparatorChar
    if (-not $candidateFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside the allowed folder: $candidateFull"
    }
    return $candidateFull
}

function Assert-NoReparseTree([string]$Root) {
    $rootFull = Get-FullPath $Root
    $rootItem = Get-Item -LiteralPath $rootFull -Force
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse point refused: $rootFull"
    }
    if (-not $rootItem.PSIsContainer) {
        return
    }

    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($rootFull)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        foreach ($item in @(Get-ChildItem -LiteralPath $current -Force)) {
            [void](Assert-ChildPath -Root $rootFull -Candidate $item.FullName -AllowRoot)
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse point refused: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                $pending.Push($item.FullName)
            }
        }
    }
}

function Assert-RepositoryName([string]$Value) {
    if ($Value -cnotmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
        throw "Invalid GitHub repository name: $Value"
    }
}

function Assert-GitHubArchiveUri([string]$Value) {
    try {
        $uri = [Uri]$Value
    } catch {
        throw "Invalid archive URI."
    }
    if ($uri.Scheme -cne 'https' -or $uri.Host -cne 'codeload.github.com' -or
            -not $uri.IsDefaultPort -or $uri.UserInfo) {
        throw 'Downloads are restricted to HTTPS on codeload.github.com.'
    }
    return $uri.AbsoluteUri
}

function Get-ReleaseMetadata([string]$Repo, [string]$MetadataPath) {
    if ($MetadataPath) {
        $resolved = (Resolve-Path -LiteralPath $MetadataPath).Path
        return (Get-Content -LiteralPath $resolved -Raw | ConvertFrom-Json)
    }

    $apiUri = "https://api.github.com/repos/$Repo/releases/latest"
    try {
        return Invoke-RestMethod -Method Get -Uri $apiUri -UseBasicParsing `
            -MaximumRedirection 0 -Headers @{
                'User-Agent' = $script:UserAgent
                'Accept' = 'application/vnd.github+json'
            }
    } catch {
        $status = $null
        if ($_.Exception.Response) {
            try { $status = [int]$_.Exception.Response.StatusCode } catch { $status = $null }
        }
        if ($status -eq 404) {
            throw "No stable GitHub Release is published for $Repo. No fallback to main was performed."
        }
        throw "Unable to read the latest stable GitHub Release ($($_.Exception.Message)). No fallback to main was performed."
    }
}

function Get-ImmutableCommit([string]$Repo, [string]$Reference) {
    if ($TestCommit) {
        return $TestCommit.ToLowerInvariant()
    }
    $escapedReference = [Uri]::EscapeDataString($Reference)
    $apiUri = "https://api.github.com/repos/$Repo/commits/$escapedReference"
    try {
        $metadata = Invoke-RestMethod -Method Get -Uri $apiUri -UseBasicParsing `
            -MaximumRedirection 0 -Headers @{
                'User-Agent' = $script:UserAgent
                'Accept' = 'application/vnd.github+json'
            }
    } catch {
        throw "Unable to resolve the immutable commit '$Reference': $($_.Exception.Message)"
    }
    $shaProperty = if ($null -ne $metadata) {
        $metadata.PSObject.Properties['sha']
    } else {
        $null
    }
    $sha = if ($null -ne $shaProperty) { [string]$shaProperty.Value } else { '' }
    if ($sha -cnotmatch '^[0-9a-fA-F]{40}$') {
        throw "GitHub did not return a valid immutable commit for '$Reference'."
    }
    return $sha.ToLowerInvariant()
}

function Resolve-ArchiveSource(
        [string]$SelectedChannel,
        [string]$Repo,
        [string]$LocalArchive,
        [string]$OverrideUri,
        [string]$MetadataPath) {
    Assert-RepositoryName $Repo

    $tag = 'main'
    $reference = 'main'
    if ($SelectedChannel -eq 'stable') {
        $release = Get-ReleaseMetadata -Repo $Repo -MetadataPath $MetadataPath
        $tagProperty = if ($null -ne $release) {
            $release.PSObject.Properties['tag_name']
        } else {
            $null
        }
        $tag = if ($null -ne $tagProperty) { [string]$tagProperty.Value } else { '' }
        if (-not $tag -or $tag -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
            throw "No usable stable GitHub Release is published for $Repo. No fallback to main was performed."
        }
        $reference = $tag
    }
    if ($TestMode -and ($LocalArchive -or $OverrideUri -or $GitRemote) -and
            -not $TestCommit) {
        throw 'TestCommit is required to bind any injection to an immutable commit.'
    }
    $commit = Get-ImmutableCommit -Repo $Repo -Reference $reference

    if ($LocalArchive) {
        $resolved = (Resolve-Path -LiteralPath $LocalArchive).Path
        if (-not [IO.File]::Exists($resolved)) {
            throw "Local archive not found: $resolved"
        }
        return [pscustomobject]@{
            Kind = 'file'; Value = $resolved; Tag = $tag
            Channel = $SelectedChannel; Commit = $commit
        }
    }
    if ($OverrideUri) {
        return [pscustomobject]@{
            Kind = 'uri'; Value = (Assert-GitHubArchiveUri $OverrideUri)
            Tag = $tag; Channel = $SelectedChannel; Commit = $commit
        }
    }
    return [pscustomobject]@{
        Kind = 'uri'
        Value = "https://codeload.github.com/$Repo/zip/$commit"
        Tag = $tag
        Channel = $SelectedChannel
        Commit = $commit
    }
}

function New-DownloadTempDirectory {
    $base = Get-FullPath ([IO.Path]::GetTempPath())
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        $leaf = 'lora-dataset-studio-update-' + [Guid]::NewGuid().ToString('N')
        $candidate = Join-Path $base $leaf
        if (-not (Test-Path -LiteralPath $candidate)) {
            [void][IO.Directory]::CreateDirectory($candidate)
            return (Get-FullPath $candidate)
        }
    }
    throw 'Unable to create a unique temporary folder.'
}

function Remove-SafeDownloadTemp([string]$Path) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return }
    $full = Get-FullPath $Path
    $base = (Get-FullPath ([IO.Path]::GetTempPath())).TrimEnd('\', '/')
    $parent = (Split-Path -Parent $full).TrimEnd('\', '/')
    $leaf = Split-Path -Leaf $full
    if (-not [string]::Equals($base, $parent, [StringComparison]::OrdinalIgnoreCase) -or
            $leaf -cnotmatch '^lora-dataset-studio-update-[0-9a-f]{32}$') {
        throw "Temporary cleanup refused for an unexpected path: $full"
    }
    $item = Get-Item -LiteralPath $full -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Temporary cleanup refused for a reparse point: $full"
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

function Assert-AvailableSpace([string]$Path, [long]$RequiredBytes) {
    $required = if ($TestRequiredFreeBytes) {
        $TestRequiredFreeBytes
    } else {
        $RequiredBytes
    }
    try {
        $root = [IO.Path]::GetPathRoot((Get-FullPath $Path))
        $drive = New-Object IO.DriveInfo($root)
        [long]$available = $drive.AvailableFreeSpace
    } catch {
        throw "Unable to check available disk space for '$Path'."
    }
    if ($available -lt $required) {
        throw "Insufficient disk space: $available bytes available, $required required."
    }
}

function Copy-StreamLimited(
        [IO.Stream]$InputStream, [IO.Stream]$OutputStream,
        [long]$MaximumBytes, [string]$Label) {
    [long]$total = 0
    $buffer = New-Object byte[] 65536
    while (($read = $InputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
        $total += [long]$read
        if ($total -gt $MaximumBytes) {
            throw "$Label exceeds the limit of $MaximumBytes bytes."
        }
        $OutputStream.Write($buffer, 0, $read)
    }
    return $total
}

function Copy-OrDownloadArchive($Source, [string]$Destination) {
    Assert-AvailableSpace -Path (Split-Path -Parent $Destination) `
        -RequiredBytes ($script:MaxArchiveBytes + $script:DiskReserveBytes)
    $output = $null
    $input = $null
    $response = $null
    try {
        $output = New-Object IO.FileStream(
            $Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
            [IO.FileShare]::None, 65536, [IO.FileOptions]::WriteThrough)
        if ($Source.Kind -eq 'file') {
            $sourceItem = Get-Item -LiteralPath $Source.Value -Force
            if ($sourceItem.PSIsContainer -or
                    ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Local archive refused (folder or reparse point): $($Source.Value)"
            }
            if ([long]$sourceItem.Length -gt $script:MaxArchiveBytes) {
                throw "Local archive exceeds the limit of $($script:MaxArchiveBytes) bytes."
            }
            $input = [IO.File]::Open(
                $sourceItem.FullName, [IO.FileMode]::Open,
                [IO.FileAccess]::Read, [IO.FileShare]::Read)
        } else {
            $safeUri = Assert-GitHubArchiveUri ([string]$Source.Value)
            Write-Step "Downloading from GitHub ($($Source.Channel): $($Source.Commit))"
            $request = [Net.HttpWebRequest][Net.WebRequest]::Create($safeUri)
            $request.Method = 'GET'
            $request.AllowAutoRedirect = $false
            $request.UserAgent = $script:UserAgent
            $request.Accept = 'application/zip'
            $response = [Net.HttpWebResponse]$request.GetResponse()
            if ([int]$response.StatusCode -ne 200) {
                throw "Download refused: HTTP $([int]$response.StatusCode)."
            }
            if ($response.ContentLength -gt $script:MaxArchiveBytes) {
                throw "Remote archive reports a size above the limit of $($script:MaxArchiveBytes) bytes."
            }
            $input = $response.GetResponseStream()
        }
        [void](Copy-StreamLimited -InputStream $input -OutputStream $output `
            -MaximumBytes $script:MaxArchiveBytes -Label 'Downloaded archive')
        $output.Flush($true)
    } catch {
        if ($output) { $output.Dispose(); $output = $null }
        if (Test-Path -LiteralPath $Destination) {
            Remove-Item -LiteralPath $Destination -Force
        }
        throw
    } finally {
        if ($input) { $input.Dispose() }
        if ($response) { $response.Dispose() }
        if ($output) { $output.Dispose() }
    }
}

function Copy-ZipEntryStreaming(
        $Entry, [string]$Destination, [ref]$GlobalBytes) {
    $input = $null
    $output = $null
    [long]$entryBytes = 0
    try {
        $input = $Entry.Open()
        $output = New-Object IO.FileStream(
            $Destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
            [IO.FileShare]::None, 65536, [IO.FileOptions]::WriteThrough)
        $buffer = New-Object byte[] 65536
        while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $entryBytes += [long]$read
            $GlobalBytes.Value += [long]$read
            if ($entryBytes -gt $script:MaxEntryBytes) {
                throw "ZIP entry '$($Entry.FullName)' exceeds the actual per-entry limit."
            }
            if ($GlobalBytes.Value -gt $script:MaxExpandedBytes) {
                throw 'ZIP archive exceeds the actual global expanded-size limit.'
            }
            $output.Write($buffer, 0, $read)
        }
        $output.Flush($true)
    } finally {
        if ($input) { $input.Dispose() }
        if ($output) { $output.Dispose() }
    }
}

function Expand-AndValidateArchive([string]$ZipPath, [string]$TempRoot) {
    Assert-AvailableSpace -Path $TempRoot -RequiredBytes (
        $script:MaxArchiveBytes + $script:MaxExpandedBytes +
        $script:DiskReserveBytes)
    $zipItem = Get-Item -LiteralPath $ZipPath -Force
    if ([long]$zipItem.Length -gt $script:MaxArchiveBytes) {
        throw 'ZIP archive above the download limit.'
    }
    $extract = Join-Path $TempRoot 'extracted'
    [void][IO.Directory]::CreateDirectory($extract)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $stream = [IO.File]::Open(
        $ZipPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $zip = New-Object IO.Compression.ZipArchive(
            $stream, [IO.Compression.ZipArchiveMode]::Read, $false)
        try {
            if ($zip.Entries.Count -eq 0) { throw 'Empty ZIP archive.' }
            if ($zip.Entries.Count -gt $script:MaxArchiveEntries) {
                throw "ZIP archive too large ($($zip.Entries.Count) entries)."
            }
            $roots = New-Object 'System.Collections.Generic.HashSet[string]' (
                [StringComparer]::OrdinalIgnoreCase)
            $seen = New-Object 'System.Collections.Generic.HashSet[string]' (
                [StringComparer]::OrdinalIgnoreCase)
            $descriptors = @()
            foreach ($entry in $zip.Entries) {
                $name = ([string]$entry.FullName).Replace('\', '/')
                if (-not $name -or $name.StartsWith('/') -or
                        $name -match '^[A-Za-z]:' -or $name.IndexOf([char]0) -ge 0) {
                    throw "Absolute or empty ZIP path refused: $name"
                }
                $trimmed = $name.TrimEnd('/')
                if (-not $trimmed) { throw 'Invalid root ZIP entry.' }
                $segments = @($trimmed.Split('/'))
                foreach ($segment in $segments) {
                    $reservedDevice = $segment -match (
                        '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)')
                    if (-not $segment -or $segment -eq '.' -or $segment -eq '..' -or
                            $segment -match '[<>:"|?*\x00-\x1F]' -or
                            $segment.EndsWith('.') -or $segment.EndsWith(' ') -or
                            $segment.Length -gt 255 -or $reservedDevice) {
                        throw "ZIP traversal/segment refused: $name"
                    }
                }
                if ($segments.Count -eq 1 -and -not $name.EndsWith('/')) {
                    throw "The archive must have a single folder root: $name"
                }
                [void]$roots.Add($segments[0])
                if (-not $seen.Add($trimmed)) {
                    throw "Duplicate ZIP entry (case included): $name"
                }
                [long]$attrs = $entry.ExternalAttributes
                if ($attrs -lt 0) { $attrs += 4294967296 }
                $unixType = (($attrs -shr 16) -band 61440)
                $dosAttrs = ($attrs -band 65535)
                if (($unixType -ne 0 -and $unixType -ne 16384 -and
                        $unixType -ne 32768) -or ($dosAttrs -band 1024) -ne 0) {
                    throw "ZIP link or special type refused: $name"
                }
                $relative = [string]::Join(
                    [IO.Path]::DirectorySeparatorChar, $segments)
                $destination = Assert-ChildPath -Root $extract `
                    -Candidate (Join-Path $extract $relative)
                $descriptors += [pscustomobject]@{
                    Entry = $entry
                    Destination = $destination
                    IsDirectory = $name.EndsWith('/')
                }
            }
            if ($roots.Count -ne 1) {
                throw "The archive must contain exactly one folder root (found: $($roots.Count))."
            }
            Write-Step 'Streaming validation and extraction of the archive'
            [long]$expandedBytes = 0
            foreach ($descriptor in $descriptors) {
                if ($descriptor.IsDirectory) {
                    [void][IO.Directory]::CreateDirectory($descriptor.Destination)
                    continue
                }
                $parent = Split-Path -Parent $descriptor.Destination
                [void][IO.Directory]::CreateDirectory($parent)
                Copy-ZipEntryStreaming -Entry $descriptor.Entry `
                    -Destination $descriptor.Destination -GlobalBytes ([ref]$expandedBytes)
            }
        } finally {
            $zip.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    Assert-NoReparseTree $extract

    $top = @(Get-ChildItem -LiteralPath $extract -Force)
    if ($top.Count -ne 1 -or -not $top[0].PSIsContainer) {
        throw "The extracted archive does not have a single folder root."
    }
    $stagedRoot = $top[0].FullName
    foreach ($relative in $script:RequiredSentinels) {
        $path = Join-Path $stagedRoot ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
        if (-not [IO.File]::Exists($path)) {
            throw "Invalid archive: missing sentinel '$relative'."
        }
        $item = Get-Item -LiteralPath $path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Invalid archive: linked/reparse sentinel '$relative'."
        }
    }
    $hasUnifiedLauncher = [IO.File]::Exists((Join-Path $stagedRoot 'start-docker.bat'))
    $hasLegacyLauncher = [IO.File]::Exists((Join-Path $stagedRoot 'start-docker-gpu.bat'))
    if (-not $hasUnifiedLauncher -and -not $hasLegacyLauncher) {
        throw "Invalid archive: missing launcher ('start-docker.bat' or 'start-docker-gpu.bat')."
    }
    return $stagedRoot
}

function Get-CodeItems([string]$StagedRoot) {
    $items = @()
    foreach ($item in @(Get-ChildItem -LiteralPath $StagedRoot -Force)) {
        if (Test-ProtectedTopLevel $item.Name) {
            Write-Host "    Local state ignored/protected: $($item.Name)"
            continue
        }
        if ($item.Name -like '.lds-update-*') {
            throw "Reserved name refused in the archive: $($item.Name)"
        }
        $items += $item
    }
    if ($items.Count -eq 0) { throw "Invalid archive: no code files to install." }
    return @($items | Sort-Object Name)
}

function Get-ActualItemBytes($Items) {
    [long]$total = 0
    foreach ($item in @($Items)) {
        Assert-NoReparseTree $item.FullName
        $files = if ($item.PSIsContainer) {
            @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -File)
        } else {
            @($item)
        }
        foreach ($file in $files) {
            [long]$length = $file.Length
            if ($length -lt 0 -or $total -gt ([long]::MaxValue - $length)) {
                throw 'Invalid or out-of-bounds size for extracted code.'
            }
            $total += $length
        }
    }
    return $total
}

function Assert-LiveDestinationsSafe([string]$Root, $Items) {
    $rootItem = Get-Item -LiteralPath $Root -Force
    if (-not $rootItem.PSIsContainer -or
            ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The installation folder is missing or is a reparse point: $Root"
    }
    foreach ($item in $Items) {
        $destination = Assert-ChildPath -Root $Root -Candidate (Join-Path $Root $item.Name)
        if (Test-Path -LiteralPath $destination) {
            Assert-NoReparseTree $destination
        }
    }
}

function Assert-ExistingInstallation([string]$Root) {
    $rootFull = Get-FullPath $Root
    $volumeRoot = (Get-FullPath ([IO.Path]::GetPathRoot($rootFull))).TrimEnd('\', '/')
    if ([string]::Equals(
            $rootFull.TrimEnd('\', '/'), $volumeRoot,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "A volume root cannot be updated: $rootFull"
    }
    $rootItem = Get-Item -LiteralPath $rootFull -Force
    if (-not $rootItem.PSIsContainer -or
            ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Installation folder refused (missing or reparse point): $rootFull"
    }
    foreach ($relative in @('docker-compose.gpu.yml', 'backend/run.py')) {
        if (-not [IO.File]::Exists((Join-Path $rootFull $relative))) {
            throw "This folder does not look like an LDS Docker installation: '$relative' is missing."
        }
    }
    if (-not [IO.File]::Exists((Join-Path $rootFull 'start-docker.bat')) -and
            -not [IO.File]::Exists((Join-Path $rootFull 'start-docker-gpu.bat'))) {
        throw 'This folder does not look like an LDS Docker installation: launcher missing.'
    }
    $gitMarker = Join-Path $rootFull '.git'
    if (Test-Path -LiteralPath $gitMarker) {
        $gitItem = Get-Item -LiteralPath $gitMarker -Force
        if (($gitItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'The .git marker cannot be a reparse point.'
        }
    }
}

function Test-UpdaterInternalRelative([string]$Relative) {
    $normalized = $Relative.Replace('\', '/')
    $top = ($normalized -split '/', 2)[0]
    return (
        [string]::Equals(
            $top, '.lds-update.lock', [StringComparison]::OrdinalIgnoreCase) -or
        $top -like '.lds-update-*')
}

function Enter-UpdateLock([string]$Root) {
    $path = Join-Path $Root '.lds-update.lock'
    if (Test-Path -LiteralPath $path) {
        $existing = Get-Item -LiteralPath $path -Force
        if ($existing.PSIsContainer -or
                ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Updater lock refused (folder or reparse point): $path"
        }
    }
    try {
        $stream = New-Object IO.FileStream(
            $path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough)
    } catch {
        throw "Another update is already using this installation (exclusive lock unavailable)."
    }
    try {
        $content = "pid=$PID utc=$([DateTime]::UtcNow.ToString('o'))"
        $bytes = [Text.Encoding]::ASCII.GetBytes($content)
        $stream.SetLength(0)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } catch {
        $stream.Dispose()
        throw
    }
    return [pscustomobject]@{ Root = $Root; Path = $path; Stream = $stream }
}

function Exit-UpdateLock($Lock) {
    if (-not $Lock) { return }
    try {
        if ($Lock.Stream) { $Lock.Stream.Dispose() }
    } finally {
        try {
            $expected = Join-Path $Lock.Root '.lds-update.lock'
            if ([string]::Equals(
                    (Get-FullPath $Lock.Path), (Get-FullPath $expected),
                    [StringComparison]::OrdinalIgnoreCase) -and
                    (Test-Path -LiteralPath $Lock.Path)) {
                $item = Get-Item -LiteralPath $Lock.Path -Force
                if (-not $item.PSIsContainer -and
                        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
                    Remove-Item -LiteralPath $Lock.Path -Force
                }
            }
        } catch {
            Write-Warning "Deferred lock cleanup: $($_.Exception.Message)"
        }
    }
}

function Assert-ExactTransactionPath(
        [string]$Path, [string]$Root, [string]$ExpectedLeaf) {
    $full = Get-FullPath $Path
    $rootFull = (Get-FullPath $Root).TrimEnd('\', '/')
    $parent = (Split-Path -Parent $full).TrimEnd('\', '/')
    if (-not [string]::Equals($rootFull, $parent, [StringComparison]::OrdinalIgnoreCase) -or
            -not [string]::Equals((Split-Path -Leaf $full), $ExpectedLeaf,
                [StringComparison]::Ordinal)) {
        throw "Unexpected transaction path refused: $full"
    }
    return $full
}

function Remove-ExactTransactionPath(
        [string]$Path, [string]$Root, [string]$ExpectedLeaf) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $full = Assert-ExactTransactionPath -Path $Path -Root $Root -ExpectedLeaf $ExpectedLeaf
    $item = Get-Item -LiteralPath $full -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Cleanup refused for a reparse point: $full"
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

function Write-DurableUtf8File([string]$Path, [string]$TempPath, [string]$Text) {
    $encoding = New-Object Text.UTF8Encoding($false)
    $bytes = $encoding.GetBytes($Text)
    if (-not (Test-Path -LiteralPath $Path)) {
        $stream = New-Object IO.FileStream(
            $Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
            [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough)
        $complete = $false
        try {
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush($true)
            $complete = $true
        } finally {
            $stream.Dispose()
            if (-not $complete) { [IO.File]::Delete($Path) }
        }
        return
    }
    $replaceBackupPath = $TempPath + '.bak'
    foreach ($stale in @($TempPath, $replaceBackupPath)) {
        if (Test-Path -LiteralPath $stale) {
            Remove-Item -LiteralPath $stale -Force
        }
    }
    $tempStream = New-Object IO.FileStream(
        $TempPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
        [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough)
    try {
        $tempStream.Write($bytes, 0, $bytes.Length)
        $tempStream.Flush($true)
    } finally {
        $tempStream.Dispose()
    }
    # Windows PowerShell 5.1 does not bind File.Replace reliably when its
    # backup argument is $null. Keep an explicit same-volume backup instead.
    [IO.File]::Replace($TempPath, $Path, $replaceBackupPath, $true)
    [IO.File]::Delete($replaceBackupPath)
}

function Write-CodeTransactionJournal($Transaction, [string]$Phase) {
    $Transaction.Phase = $Phase
    $journal = [ordered]@{
        schema = 'lds-docker-code-update/v2'
        id = $Transaction.Id
        root = $Transaction.Root
        phase = $Phase
        stage_leaf = $Transaction.StageLeaf
        backup_leaf = $Transaction.BackupLeaf
        failed_leaf = $Transaction.FailedLeaf
        journal_leaf = $Transaction.JournalLeaf
        items = @($Transaction.Records | ForEach-Object {
            [ordered]@{
                name = $_.Name
                had_original = [bool]$_.HadOriginal
                has_new = [bool]$_.HasNew
            }
        })
    } | ConvertTo-Json -Depth 6 -Compress
    Write-DurableUtf8File -Path $Transaction.JournalPath `
        -TempPath $Transaction.JournalTempPath -Text $journal
}

function Remove-TransactionArtifactsBestEffort($Transaction) {
    $artifacts = @(
        [pscustomobject]@{ Path = $Transaction.FailedRoot; Leaf = $Transaction.FailedLeaf },
        [pscustomobject]@{ Path = $Transaction.StageRoot; Leaf = $Transaction.StageLeaf },
        [pscustomobject]@{ Path = $Transaction.BackupRoot; Leaf = $Transaction.BackupLeaf },
        [pscustomobject]@{
            Path = $Transaction.JournalTempPath; Leaf = $Transaction.JournalTempLeaf
        },
        [pscustomobject]@{
            Path = $Transaction.JournalReplaceBackupPath
            Leaf = $Transaction.JournalReplaceBackupLeaf
        },
        [pscustomobject]@{ Path = $Transaction.JournalPath; Leaf = $Transaction.JournalLeaf }
    )
    foreach ($artifact in $artifacts) {
        try {
            Remove-ExactTransactionPath -Path $artifact.Path `
                -Root $Transaction.Root -ExpectedLeaf $artifact.Leaf
        } catch {
            Write-Warning "Deferred cleanup for '$($artifact.Path)': $($_.Exception.Message)"
        }
    }
}

function Assert-CodeTransactionArtifactsSafe($Transaction) {
    foreach ($artifact in @(
            [pscustomobject]@{ Path = $Transaction.StageRoot; Directory = $true },
            [pscustomobject]@{ Path = $Transaction.BackupRoot; Directory = $true },
            [pscustomobject]@{ Path = $Transaction.FailedRoot; Directory = $true },
            [pscustomobject]@{ Path = $Transaction.JournalPath; Directory = $false },
            [pscustomobject]@{ Path = $Transaction.JournalTempPath; Directory = $false },
            [pscustomobject]@{
                Path = $Transaction.JournalReplaceBackupPath; Directory = $false
            })) {
        if (-not (Test-Path -LiteralPath $artifact.Path)) { continue }
        $item = Get-Item -LiteralPath $artifact.Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
                [bool]$item.PSIsContainer -ne [bool]$artifact.Directory) {
            throw "Transaction artifact refused (type or reparse point): $($artifact.Path)"
        }
        if ($artifact.Directory) { Assert-NoReparseTree $artifact.Path }
    }
}

function Undo-CodeOverlay($Transaction) {
    if (-not $Transaction -or -not $Transaction.Active) { return }
    Assert-CodeTransactionArtifactsSafe $Transaction
    Write-Step 'Rolling back previous code'
    Write-CodeTransactionJournal -Transaction $Transaction -Phase 'rolling-back'
    $errors = @()
    if (-not (Test-Path -LiteralPath $Transaction.FailedRoot)) {
        [void][IO.Directory]::CreateDirectory($Transaction.FailedRoot)
    }
    for ($index = $Transaction.Records.Count - 1; $index -ge 0; $index--) {
        $record = $Transaction.Records[$index]
        $destination = Join-Path $Transaction.Root $record.Name
        $backup = Join-Path $Transaction.BackupRoot $record.Name
        $stage = Join-Path $Transaction.StageRoot $record.Name
        try {
            if (Test-Path -LiteralPath $backup) {
                if (Test-Path -LiteralPath $destination) {
                    $failedName = $record.Name + '.' + [Guid]::NewGuid().ToString('N')
                    Move-Item -LiteralPath $destination `
                        -Destination (Join-Path $Transaction.FailedRoot $failedName)
                }
                Move-Item -LiteralPath $backup -Destination $destination
            } elseif ($record.HadOriginal) {
                if (-not (Test-Path -LiteralPath $destination)) {
                    throw "Missing original and backup for $($record.Name)"
                }
            } elseif ($record.HasNew -and
                    -not (Test-Path -LiteralPath $stage) -and
                    (Test-Path -LiteralPath $destination)) {
                $failedName = $record.Name + '.' + [Guid]::NewGuid().ToString('N')
                Move-Item -LiteralPath $destination `
                    -Destination (Join-Path $Transaction.FailedRoot $failedName)
            }
        } catch {
            $errors += "$($record.Name): $($_.Exception.Message)"
        }
    }
    if ($errors.Count -gt 0) {
        throw "Incomplete rollback. Journal '$($Transaction.JournalPath)'. $($errors -join ' | ')"
    }
    Write-CodeTransactionJournal -Transaction $Transaction -Phase 'rolled-back'
    $Transaction.Active = $false
    Remove-TransactionArtifactsBestEffort $Transaction
}

function Get-ExistingCodeItems([string]$Root) {
    $items = @()
    foreach ($item in @(Get-ChildItem -LiteralPath $Root -Force)) {
        if (Test-ProtectedTopLevel $item.Name) { continue }
        if (Test-UpdaterInternalRelative $item.Name) { continue }
        Assert-NoReparseTree $item.FullName
        $items += $item
    }
    return @($items | Sort-Object Name)
}

function New-CodeTransaction([string]$Root, $Records) {
    $id = [Guid]::NewGuid().ToString('N')
    $stageLeaf = ".lds-update-stage-$id"
    $backupLeaf = ".lds-update-backup-$id"
    $failedLeaf = ".lds-update-failed-$id"
    $journalLeaf = ".lds-update-journal-$id.json"
    $journalTempLeaf = ".lds-update-journal-$id.tmp"
    $journalReplaceBackupLeaf = ".lds-update-journal-$id.tmp.bak"
    return [pscustomobject]@{
        Id = $id; Root = $Root; Records = @($Records); Active = $true
        Phase = 'new'
        StageLeaf = $stageLeaf; StageRoot = (Join-Path $Root $stageLeaf)
        BackupLeaf = $backupLeaf; BackupRoot = (Join-Path $Root $backupLeaf)
        FailedLeaf = $failedLeaf; FailedRoot = (Join-Path $Root $failedLeaf)
        JournalLeaf = $journalLeaf; JournalPath = (Join-Path $Root $journalLeaf)
        JournalTempLeaf = $journalTempLeaf
        JournalTempPath = (Join-Path $Root $journalTempLeaf)
        JournalReplaceBackupLeaf = $journalReplaceBackupLeaf
        JournalReplaceBackupPath = (Join-Path $Root $journalReplaceBackupLeaf)
    }
}

function Install-CodeOverlay([string]$StagedRoot, [string]$Root, $ValidatedItems) {
    $newItems = @($ValidatedItems)
    if ($newItems.Count -eq 0) { throw 'No valid code to install.' }
    Assert-LiveDestinationsSafe -Root $Root -Items $newItems
    $oldItems = @(Get-ExistingCodeItems $Root)
    $newMap = @{}
    $oldMap = @{}
    $names = New-Object 'System.Collections.Generic.HashSet[string]' (
        [StringComparer]::OrdinalIgnoreCase)
    foreach ($item in $newItems) {
        $newMap[$item.Name] = $item
        [void]$names.Add($item.Name)
    }
    foreach ($item in $oldItems) {
        $oldMap[$item.Name] = $item
        [void]$names.Add($item.Name)
    }
    $records = @()
    foreach ($name in @($names | Sort-Object)) {
        $records += [pscustomobject]@{
            Name = $name
            HadOriginal = [bool]$oldMap.ContainsKey($name)
            HasNew = [bool]$newMap.ContainsKey($name)
        }
    }
    if ($records.Count -eq 0) { throw 'No old or new code to transact.' }
    $transaction = New-CodeTransaction -Root $Root -Records $records
    # Persist intent before creating any other transaction artifact so every
    # interruption window has a discoverable recovery record.
    Write-CodeTransactionJournal -Transaction $transaction -Phase 'preparing'
    try {
        [void][IO.Directory]::CreateDirectory($transaction.StageRoot)
        [void][IO.Directory]::CreateDirectory($transaction.BackupRoot)
        Write-Step 'Preparing new code locally'
        foreach ($item in $newItems) {
            Copy-Item -LiteralPath $item.FullName `
                -Destination (Join-Path $transaction.StageRoot $item.Name) `
                -Recurse -Force
        }
        Write-CodeTransactionJournal -Transaction $transaction -Phase 'prepared'
        Write-Step 'Transactionally installing code'
        Write-CodeTransactionJournal -Transaction $transaction -Phase 'switching'
        $switched = 0
        foreach ($record in $records) {
            $source = Join-Path $transaction.StageRoot $record.Name
            $destination = Join-Path $Root $record.Name
            $backup = Join-Path $transaction.BackupRoot $record.Name
            if ($record.HadOriginal) {
                Move-Item -LiteralPath $destination -Destination $backup
            }
            if ($record.HasNew) {
                Move-Item -LiteralPath $source -Destination $destination
            }
            $switched++
            if ($TestMode -and $TestFault -eq 'overlay-after-first-switch' -and
                    $switched -eq 1) {
                [Environment]::Exit(86)
            }
        }
        Write-CodeTransactionJournal -Transaction $transaction -Phase 'switched'
        return $transaction
    } catch {
        try { Undo-CodeOverlay $transaction } catch {
            throw "Install failure, and $($_.Exception.Message)"
        }
        throw
    }
}

function Complete-CodeOverlay($Transaction) {
    if (-not $Transaction -or -not $Transaction.Active) { return }
    Write-CodeTransactionJournal -Transaction $Transaction -Phase 'committed'
    $Transaction.Active = $false
    Remove-TransactionArtifactsBestEffort $Transaction
}

function ConvertFrom-CodeTransactionJournal([string]$Root, [string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Transaction journal refused: $Path"
    }
    try {
        $data = [IO.File]::ReadAllText($item.FullName, [Text.Encoding]::UTF8) |
            ConvertFrom-Json
    } catch {
        throw "Unreadable transaction journal: $Path"
    }
    if ($data.schema -cne 'lds-docker-code-update/v2' -or
            [string]$data.id -cnotmatch '^[0-9a-f]{32}$' -or
            -not [string]::Equals(
                (Get-FullPath ([string]$data.root)), (Get-FullPath $Root),
                [StringComparison]::OrdinalIgnoreCase) -or
            [string]$data.phase -notin @(
                'preparing', 'prepared', 'switching', 'switched',
                'launching', 'rebuild-launched', 'rolling-back',
                'rolled-back', 'committed')) {
        throw "Invalid transaction journal: $Path"
    }
    $id = [string]$data.id
    $expected = @{
        stage_leaf = ".lds-update-stage-$id"
        backup_leaf = ".lds-update-backup-$id"
        failed_leaf = ".lds-update-failed-$id"
        journal_leaf = ".lds-update-journal-$id.json"
    }
    foreach ($property in $expected.Keys) {
        if ([string]$data.$property -cne $expected[$property]) {
            throw "Inconsistent transaction journal ($property): $Path"
        }
    }
    $records = @()
    foreach ($entry in @($data.items)) {
        $name = [string]$entry.name
        if (-not $name -or
                $name.IndexOfAny([char[]]@('/', '\', ':', [char]0)) -ge 0 -or
                (Test-ProtectedTopLevel $name) -or
                (Test-UpdaterInternalRelative $name)) {
            throw "Transaction name refused in the journal: '$name'"
        }
        $records += [pscustomobject]@{
            Name = $name
            HadOriginal = [bool]$entry.had_original
            HasNew = [bool]$entry.has_new
        }
    }
    if ($records.Count -eq 0) { throw "Transaction journal has no items: $Path" }
    $transaction = New-CodeTransaction -Root $Root -Records $records
    $transaction.Id = $id
    $transaction.StageLeaf = $expected.stage_leaf
    $transaction.StageRoot = Join-Path $Root $transaction.StageLeaf
    $transaction.BackupLeaf = $expected.backup_leaf
    $transaction.BackupRoot = Join-Path $Root $transaction.BackupLeaf
    $transaction.FailedLeaf = $expected.failed_leaf
    $transaction.FailedRoot = Join-Path $Root $transaction.FailedLeaf
    $transaction.JournalLeaf = $expected.journal_leaf
    $transaction.JournalPath = Join-Path $Root $transaction.JournalLeaf
    $transaction.JournalTempLeaf = ".lds-update-journal-$id.tmp"
    $transaction.JournalTempPath = Join-Path $Root $transaction.JournalTempLeaf
    $transaction.JournalReplaceBackupLeaf = ".lds-update-journal-$id.tmp.bak"
    $transaction.JournalReplaceBackupPath = Join-Path `
        $Root $transaction.JournalReplaceBackupLeaf
    $transaction.Phase = [string]$data.phase
    return $transaction
}

function Recover-PendingCodeTransaction([string]$Root) {
    $journals = @(Get-ChildItem -LiteralPath $Root -Force -File |
        Where-Object { $_.Name -match '^\.lds-update-journal-[0-9a-f]{32}\.json$' })
    if ($journals.Count -gt 1) {
        throw 'Multiple updater journals are present; automatic recovery refused.'
    }
    if ($journals.Count -eq 0) { return }
    $transaction = ConvertFrom-CodeTransactionJournal `
        -Root $Root -Path $journals[0].FullName
    Write-Step "Recovering interrupted transaction ($($transaction.Phase))"
    if ($transaction.Phase -in @('committed', 'rolled-back')) {
        $transaction.Active = $false
        Remove-TransactionArtifactsBestEffort $transaction
        return
    }
    Undo-CodeOverlay $transaction
}

function Read-LauncherModeMarker([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer -and
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        $bytes = [IO.File]::ReadAllBytes($item.FullName)
    } else {
        throw "Settings launcher refused (folder or reparse point): $Path"
    }
    foreach ($byte in $bytes) {
        if ($byte -gt 127) {
            throw 'Invalid settings launcher: strict ASCII required.'
        }
    }
    $text = [Text.Encoding]::ASCII.GetString($bytes)
    $normalized = $text.Replace("`r`n", "`n")
    if ($normalized.Contains("`r")) {
        throw 'Invalid settings launcher: isolated CR line ending.'
    }
    if ($normalized.EndsWith("`n")) {
        $normalized = $normalized.Substring(0, $normalized.Length - 1)
    }
    if ($normalized.Contains("`n") -or
            $normalized -cnotmatch '^LAST_LAUNCHER=(studio|gpu)$') {
        throw 'Invalid settings launcher: exactly one LAST_LAUNCHER=studio|gpu line is required.'
    }
    return $matches[1]
}

function Get-LauncherPreference([string]$Root, [string]$Override) {
    if ($Override) {
        $path = if ([IO.Path]::IsPathRooted($Override)) {
            $Override
        } else {
            Join-Path $Root $Override
        }
        return [pscustomobject]@{ Kind = 'override'; Path = (Get-FullPath $path) }
    }
    $settingsPath = Join-Path $Root '.docker-launch-settings'
    # No marker means the historical GPU launcher/data-docker-gpu lane. Merely
    # receiving a new launcher file in an archive never migrates that lane.
    $mode = if (Test-Path -LiteralPath $settingsPath) {
        Read-LauncherModeMarker $settingsPath
    } else {
        'gpu'
    }
    return [pscustomobject]@{
        Kind = $mode
        Path = ''
    }
}

function Assert-SafeLauncherFile(
        [string]$Path, [string]$Root, [switch]$AllowExternal) {
    $full = Get-FullPath $Path
    if (-not [IO.File]::Exists($full)) {
        throw "Required launcher not found: $full"
    }
    $item = Get-Item -LiteralPath $full -Force
    if ($item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Launcher refused (folder or reparse point): $full"
    }
    if (-not $AllowExternal) {
        $parent = (Split-Path -Parent $full).TrimEnd('\', '/')
        $rootFull = (Get-FullPath $Root).TrimEnd('\', '/')
        if (-not [string]::Equals(
                $parent, $rootFull, [StringComparison]::OrdinalIgnoreCase) -or
                [IO.Path]::GetExtension($full) -cne '.bat') {
            throw "Internal launcher outside root or refused extension: $full"
        }
    } elseif (-not $TestMode) {
        throw 'An external launcher is strictly reserved for test mode.'
    }
    return $full
}

function Resolve-PreferredLauncher(
        [string]$Root, $Preference, [switch]$Rollback) {
    if ($Preference.Kind -eq 'override') {
        return (Assert-SafeLauncherFile -Path $Preference.Path -Root $Root `
            -AllowExternal)
    }
    $name = if ($Preference.Kind -eq 'studio') {
        'start-docker.bat'
    } else {
        'start-docker-gpu.bat'
    }
    $candidate = Join-Path $Root $name
    return (Assert-SafeLauncherFile -Path $candidate -Root $Root)
}

function Assert-StagedLauncherCompatibility([string]$StagedRoot, $Preference) {
    if ($Preference.Kind -eq 'override') { return }
    $required = if ($Preference.Kind -eq 'studio') {
        'start-docker.bat'
    } else {
        'start-docker-gpu.bat'
    }
    if (-not [IO.File]::Exists((Join-Path $StagedRoot $required))) {
        throw "Archive incompatible with the $($Preference.Kind) installation: '$required' is missing."
    }
    $item = Get-Item -LiteralPath (Join-Path $StagedRoot $required) -Force
    if ($item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Incompatible archive: launcher '$required' is linked or a reparse point."
    }
}

function Invoke-RebuildLauncher([string]$Path) {
    if (-not [IO.File]::Exists($Path)) {
        Write-Host "[ERROR] Launcher not found: $Path" -ForegroundColor Red
        return 127
    }
    Write-Step 'Build/recreate without prior stop (--update-rebuild)'
    try {
        # Out-Host keeps the launcher's own stdout out of this function's return
        # value; without it the caller received an array and printed the whole
        # banner inside "(code ... )" instead of the exit code.
        & $Path '--update-rebuild' | Out-Host
        if ($null -eq $LASTEXITCODE) { return 0 }
        return [int]$LASTEXITCODE
    } catch {
        Write-Host "[ERROR] The launcher failed: $($_.Exception.Message)" -ForegroundColor Red
        return 126
    }
}

function Get-GitExecutable {
    $command = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $command) {
        $command = Get-Command git -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if ($command) { return $command.Source }
    return $null
}

function ConvertTo-NativeArgument([string]$Value) {
    if ($null -eq $Value) { return '""' }
    if ($Value.Length -gt 0 -and $Value -cnotmatch '[\s"]') { return $Value }
    # Windows CommandLineToArgvW-compatible escaping: backslashes are doubled
    # only when they precede a quote or the closing quote.
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    [int]$slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $slashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($slashes * 2) + 1)))
            [void]$builder.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) {
            [void]$builder.Append(('\' * $slashes))
            $slashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($slashes -gt 0) { [void]$builder.Append(('\' * ($slashes * 2))) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-Git([string]$Git, [string]$Root, [string[]]$Arguments) {
    # Process redirection preserves NUL delimiters and embedded newlines in
    # filenames. PowerShell's native pipeline would split those records.
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $Git
    $start.WorkingDirectory = $Root
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = [Text.Encoding]::UTF8
    $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    $start.Arguments = (@($Arguments | ForEach-Object {
        ConvertTo-NativeArgument ([string]$_)
    }) -join ' ')
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw 'git.exe failed to start.' }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        $text = (($stdout + $stderr).Trim())
        return [pscustomobject]@{
            Code = [int]$process.ExitCode
            Text = $text
            Stdout = $stdout
            Stderr = $stderr
        }
    } finally {
        $process.Dispose()
    }
}

function Get-GitNullRecords($Result) {
    if ($Result.Code -ne 0) { return @() }
    return @(([string]$Result.Stdout).Split([char]0) |
        Where-Object { $_ -ne '' })
}

function Get-GitDirtyRecords([string]$Git, [string]$Root) {
    $status = Invoke-Git -Git $Git -Root $Root -Arguments @(
        '-c', 'core.quotePath=false', 'status', '--porcelain=v1', '-z',
        '--untracked-files=all')
    if ($status.Code -ne 0) {
        throw "Unable to check Git status: $($status.Text)"
    }
    $dirty = @()
    foreach ($record in @(Get-GitNullRecords $status)) {
        if ($record.Length -ge 4) {
            $relative = $record.Substring(3)
            if (Test-UpdaterInternalRelative $relative) { continue }
        }
        $dirty += $record
    }
    return $dirty
}

function Assert-GitWorktreeClean(
        [string]$Git, [string]$Root, [string]$Moment) {
    $dirty = @(Get-GitDirtyRecords -Git $Git -Root $Root)
    if ($dirty.Count -gt 0) {
        throw ('Modified Git checkout (' + $Moment +
            '): update refused without overwriting.' +
            ' Commit or stash your changes, run git pull --ff-only,' +
            ' then run your Docker launcher with --update-rebuild.' +
            [Environment]::NewLine + ($dirty -join [Environment]::NewLine))
    }
}

function Assert-GitIgnoredStateSafe([string]$Git, [string]$Root) {
    $ignored = Invoke-Git -Git $Git -Root $Root `
        -Arguments @(
            '-c', 'core.quotePath=false', 'ls-files', '-z',
            '--others', '--ignored', '--exclude-standard')
    if ($ignored.Code -ne 0) {
        throw "Unable to inspect ignored Git files: $($ignored.Text)"
    }
    foreach ($relative in @(Get-GitNullRecords $ignored)) {
        if (-not $relative) { continue }
        $normalized = $relative.Replace('\', '/')
        $top = ($normalized -split '/', 2)[0]
        if (-not (Test-ProtectedTopLevel $top)) {
            throw "Git checkout refused: ignored file outside protected state '$relative'."
        }
    }
}

function Assert-GitTargetTreeSafe(
        [string]$Git, [string]$Root, [string]$Commit,
        [string]$Label = 'cible') {
    $tree = Invoke-Git -Git $Git -Root $Root -Arguments @(
        '-c', 'core.quotePath=false', 'ls-tree', '-r', '-z', '--full-tree', $Commit)
    if ($tree.Code -ne 0) {
        throw "Unable to inspect the target Git tree: $($tree.Text)"
    }
    foreach ($line in @(Get-GitNullRecords $tree)) {
        if (-not $line) { continue }
        if ($line -cnotmatch '^(100644|100755) blob [0-9a-f]+\t(.+)$') {
            throw "Git tree $Label refused (link, submodule or special mode): $line"
        }
        $relative = $matches[2].Replace('\', '/')
        $top = ($relative -split '/', 2)[0]
        if ((Test-ProtectedTopLevel $top) -or
                (Test-UpdaterInternalRelative $top)) {
            throw "Git tree $Label refused: protected local state tracked by Git '$relative'."
        }
    }
}

function Invoke-GitCheckoutFailClosed(
        [string]$Root, [string]$SelectedLauncher) {
    $launcherPreference = Get-LauncherPreference -Root $Root -Override $SelectedLauncher
    $launcher = Resolve-PreferredLauncher -Root $Root -Preference $launcherPreference
    $git = Get-GitExecutable
    if (-not $git) {
        throw "This folder contains .git but git.exe is unavailable. No file was overwritten."
    }
    Assert-GitWorktreeClean -Git $git -Root $Root -Moment 'initial state'
    Assert-GitIgnoredStateSafe -Git $git -Root $Root
    $commitResult = Invoke-Git -Git $git -Root $Root `
        -Arguments @('rev-parse', 'HEAD^{commit}')
    if ($commitResult.Code -ne 0 -or
            $commitResult.Text -cnotmatch '^[0-9a-fA-F]{40}$') {
        throw "Unable to identify the current Git commit."
    }
    $commit = $commitResult.Text.Trim()
    Assert-GitTargetTreeSafe -Git $git -Root $Root -Commit $commit `
        -Label 'actuel'
    throw ('Git checkout detected: no Git file was modified. ' +
        'For safety, this automatic updater is reserved for ZIP installations. ' +
        'Keep your changes, run git pull --ff-only in this folder, ' +
        "then run '$launcher' --update-rebuild to rebuild the stack.")
}

function Invoke-ZipInstallUpdate(
        [string]$Root, [string]$StagedRoot, [string]$SelectedLauncher,
        $ValidatedItems) {
    $launcherPreference = Get-LauncherPreference -Root $Root -Override $SelectedLauncher
    # Resolve and validate the currently selected launcher while old code is
    # still intact. Rollback must never discover a broken launcher too late.
    [void](Resolve-PreferredLauncher -Root $Root -Preference $launcherPreference)
    Assert-StagedLauncherCompatibility -StagedRoot $StagedRoot `
        -Preference $launcherPreference
    [long]$newCodeBytes = Get-ActualItemBytes $ValidatedItems
    Assert-AvailableSpace -Path $Root `
        -RequiredBytes ($newCodeBytes + $script:DiskReserveBytes)
    $transaction = Install-CodeOverlay -StagedRoot $StagedRoot -Root $Root `
        -ValidatedItems $ValidatedItems
    Write-CodeTransactionJournal -Transaction $transaction -Phase 'launching'
    $launcher = Resolve-PreferredLauncher -Root $Root -Preference $launcherPreference
    $code = Invoke-RebuildLauncher $launcher
    if ($code -eq 0) {
        # --update-rebuild returns 0 only after the launcher has polled Docker
        # until the Studio container reports healthy; it exits non-zero when the
        # container dies, turns unhealthy or never gets there. So 0 is a health
        # confirmation and the transaction commits. Leaving it pending instead
        # would make the NEXT run recover this journal and roll a working
        # install back to the previous code.
        Write-CodeTransactionJournal -Transaction $transaction `
            -Phase 'rebuild-launched'
        Complete-CodeOverlay $transaction
        Write-Host '[OK] New version installed and confirmed healthy by the launcher.' `
            -ForegroundColor Green
        return
    }

    Write-Host "[ERROR] Build/start of the new version (code $code). Rolling back..." -ForegroundColor Red
    Undo-CodeOverlay $transaction
    $oldLauncher = Resolve-PreferredLauncher -Root $Root `
        -Preference $launcherPreference -Rollback
    $oldCode = Invoke-RebuildLauncher $oldLauncher
    if ($oldCode -ne 0) {
        throw "New version rejected; old code restored, but relaunching the old version failed (code $oldCode)."
    }
    throw "New version rejected (code $code). Old code restored and old stack relaunched."
}

function Invoke-DockerGpuUpdate {
    if (-not $InstallRoot) {
        $scriptParent = Split-Path -Parent $PSScriptRoot
        $script:InstallRoot = $scriptParent
    }
    $root = Get-FullPath $InstallRoot
    if (-not [IO.Directory]::Exists($root)) {
        throw "Installation folder not found: $root"
    }
    Assert-ExistingInstallation $root
    Assert-RepositoryName $Repository

    $lock = $null
    $downloadTemp = ''
    try {
        $lock = Enter-UpdateLock $root
        Recover-PendingCodeTransaction $root

        if ($TestMode -and $TestFault -eq 'hold-lock') {
            [IO.File]::WriteAllText(
                (Get-FullPath $TestSignalPath), 'locked',
                (New-Object Text.UTF8Encoding($false)))
            Start-Sleep -Milliseconds $TestHoldMilliseconds
            throw 'TestFault hold-lock finished after holding the lock.'
        }

        Write-Host "Channel: $Channel" -ForegroundColor Yellow
        if (Test-Path -LiteralPath (Join-Path $root '.git')) {
            Invoke-GitCheckoutFailClosed -Root $root `
                -SelectedLauncher $LauncherPath
            return
        }

        # Windows PowerShell 5.1 can otherwise negotiate TLS 1.0 on older machines.
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $source = Resolve-ArchiveSource -SelectedChannel $Channel `
            -Repo $Repository -LocalArchive $ArchivePath `
            -OverrideUri $ArchiveUri -MetadataPath $ReleaseMetadataPath
        Write-Host "Selected immutable commit: $($source.Commit)" -ForegroundColor Yellow
        Write-Warning ('The GitHub archive is bound to this commit via HTTPS, but ' +
            'this updater does not verify any cryptographic signature of the project.')

        $downloadTemp = New-DownloadTempDirectory
        $zipPath = Join-Path $downloadTemp 'source.zip'
        Copy-OrDownloadArchive -Source $source -Destination $zipPath
        $stagedRoot = Expand-AndValidateArchive -ZipPath $zipPath -TempRoot $downloadTemp
        $validatedItems = @(Get-CodeItems $stagedRoot)
        Assert-LiveDestinationsSafe -Root $root -Items $validatedItems

        Invoke-ZipInstallUpdate -Root $root -StagedRoot $stagedRoot `
            -SelectedLauncher $LauncherPath -ValidatedItems $validatedItems
    } finally {
        try {
            if ($downloadTemp) { Remove-SafeDownloadTemp $downloadTemp }
        } finally {
            Exit-UpdateLock $lock
        }
    }
}

function Assert-RunningFromSafeTemp {
    $runnerDirectory = Get-FullPath (Split-Path -Parent $PSCommandPath)
    $tempBase = (Get-FullPath ([IO.Path]::GetTempPath())).TrimEnd('\', '/')
    $runnerParent = (Split-Path -Parent $runnerDirectory).TrimEnd('\', '/')
    $runnerLeaf = Split-Path -Leaf $runnerDirectory
    if (-not [string]::Equals(
            $tempBase, $runnerParent, [StringComparison]::OrdinalIgnoreCase) -or
            $runnerLeaf -cnotmatch '^lora-dataset-studio-update-[0-9a-f]{32}$') {
        throw 'The internal RunningFromTemp mode is refused outside the updater unique temp folder.'
    }
    $runnerItem = Get-Item -LiteralPath $runnerDirectory -Force
    if (($runnerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'The temporary runner cannot be a reparse point.'
    }
}

function Invoke-SelfBootstrap {
    $root = if ($InstallRoot) {
        Get-FullPath $InstallRoot
    } else {
        Get-FullPath (Split-Path -Parent $PSScriptRoot)
    }
    $runnerTemp = New-DownloadTempDirectory
    try {
        $runner = Join-Path $runnerTemp 'update-docker-gpu.runner.ps1'
        Copy-Item -LiteralPath $PSCommandPath -Destination $runner

        $windowsPowerShell = Join-Path $env:SystemRoot (
            'System32\WindowsPowerShell\v1.0\powershell.exe')
        if (-not [IO.File]::Exists($windowsPowerShell)) {
            $command = Get-Command powershell.exe -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if (-not $command) {
                throw 'Windows PowerShell 5.1 was not found.'
            }
            $windowsPowerShell = $command.Source
        }

        $childArguments = @(
            '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
            '-File', $runner,
            '-RunningFromTemp',
            '-Channel', $Channel,
            '-InstallRoot', $root,
            '-Repository', $Repository
        )
        if ($TestMode) {
            $childArguments += '-TestMode'
            if ($ArchivePath) {
                $childArguments += @('-ArchivePath', (Get-FullPath $ArchivePath))
            }
            if ($ArchiveUri) { $childArguments += @('-ArchiveUri', $ArchiveUri) }
            if ($ReleaseMetadataPath) {
                $childArguments += @(
                    '-ReleaseMetadataPath', (Get-FullPath $ReleaseMetadataPath))
            }
            if ($LauncherPath) { $childArguments += @('-LauncherPath', $LauncherPath) }
            if ($GitRemote) { $childArguments += @('-GitRemote', $GitRemote) }
            if ($TestCommit) { $childArguments += @('-TestCommit', $TestCommit) }
            if ($TestFault) { $childArguments += @('-TestFault', $TestFault) }
            if ($TestSignalPath) {
                $childArguments += @('-TestSignalPath', (Get-FullPath $TestSignalPath))
            }
            if ($TestHoldMilliseconds) {
                $childArguments += @('-TestHoldMilliseconds', $TestHoldMilliseconds)
            }
            if ($TestMaxArchiveBytes) {
                $childArguments += @('-TestMaxArchiveBytes', $TestMaxArchiveBytes)
            }
            if ($TestMaxEntryBytes) {
                $childArguments += @('-TestMaxEntryBytes', $TestMaxEntryBytes)
            }
            if ($TestMaxExpandedBytes) {
                $childArguments += @('-TestMaxExpandedBytes', $TestMaxExpandedBytes)
            }
            if ($TestRequiredFreeBytes) {
                $childArguments += @('-TestRequiredFreeBytes', $TestRequiredFreeBytes)
            }
        }

        # Consume the child success stream explicitly so the function returns
        # exactly one value: its numeric exit code. Otherwise PowerShell bundles
        # every child log line plus the integer into an Object[], and exit can
        # coerce that array to success even when the child returned non-zero.
        & $windowsPowerShell @childArguments | ForEach-Object { Write-Host $_ }
        if ($null -eq $LASTEXITCODE) { return 1 }
        return [int]$LASTEXITCODE
    } finally {
        Remove-SafeDownloadTemp $runnerTemp
    }
}

try {
    Assert-TestModeConfiguration
} catch {
    Write-Host ''
    Write-Host "[ERROR] Updater configuration refused: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

if (-not $RunningFromTemp) {
    try {
        exit (Invoke-SelfBootstrap)
    } catch {
        Write-Host ''
        Write-Host "[ERROR] Unable to bootstrap updater: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

try {
    Assert-RunningFromSafeTemp
    Invoke-DockerGpuUpdate
    exit 0
} catch {
    Write-Host ''
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
