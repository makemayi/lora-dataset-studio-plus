[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Read', 'Wait')]
    [string] $Action,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ConfigPath,

    [ValidateRange(1, 3600)]
    [int] $TimeoutSeconds = 900,

    [ValidateRange(1, 30)]
    [int] $PollSeconds = 2
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8NoBom

function Get-DeploymentMode {
    param([string] $LiteralPath)

    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        return @{ State = 'VALID'; Mode = 'unset' }
    }

    try {
        $root = Get-Content -LiteralPath $LiteralPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
    }
    catch {
        return @{ State = 'INVALID'; Mode = 'unset' }
    }

    if ($null -eq $root -or
        $root.GetType() -ne [System.Management.Automation.PSCustomObject]) {
        return @{ State = 'INVALID'; Mode = 'unset' }
    }

    $ollamaProperty = $root.PSObject.Properties['ollama']
    if ($null -eq $ollamaProperty -or $null -eq $ollamaProperty.Value) {
        return @{ State = 'VALID'; Mode = 'unset' }
    }
    $ollama = $ollamaProperty.Value
    if ($ollama.GetType() -ne [System.Management.Automation.PSCustomObject]) {
        return @{ State = 'INVALID'; Mode = 'unset' }
    }

    $modeProperty = $ollama.PSObject.Properties['deployment_mode']
    if ($null -eq $modeProperty -or $null -eq $modeProperty.Value) {
        return @{ State = 'VALID'; Mode = 'unset' }
    }
    if ($modeProperty.Value -isnot [string]) {
        return @{ State = 'INVALID'; Mode = 'unset' }
    }

    $mode = $modeProperty.Value.Trim().ToLowerInvariant()
    if (-not $mode -or $mode -eq 'unconfigured') {
        return @{ State = 'VALID'; Mode = 'unset' }
    }
    if ($mode -notin @('none', 'host', 'docker')) {
        return @{ State = 'INVALID'; Mode = 'unset' }
    }
    return @{ State = 'VALID'; Mode = $mode }
}

function Write-Result {
    param([string] $State, [string] $Mode)

    [Console]::Out.WriteLine(('STATE={0}' -f $State))
    [Console]::Out.WriteLine(('MODE={0}' -f $Mode))
}

$fullPath = [System.IO.Path]::GetFullPath($ConfigPath)
if ($Action -eq 'Read') {
    $result = Get-DeploymentMode -LiteralPath $fullPath
    Write-Result -State $result.State -Mode $result.Mode
    exit 0
}

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
do {
    $result = Get-DeploymentMode -LiteralPath $fullPath
    if ($result.State -eq 'INVALID') {
        Write-Result -State 'INVALID' -Mode 'unset'
        exit 0
    }
    if ($result.Mode -ne 'unset') {
        Write-Result -State 'VALID' -Mode $result.Mode
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
} while ([DateTime]::UtcNow -lt $deadline)

Write-Result -State 'TIMEOUT' -Mode 'unset'
exit 0
