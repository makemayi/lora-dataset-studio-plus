[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $DockerExe,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ContainerName,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ExpectedProject,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ExpectedService,

    [ValidateSet('', 'studio', 'external', 'gpu')]
    [string] $ExpectedMode = '',

    [ValidateSet('', 'none', 'host', 'docker')]
    [string] $ExpectedOllamaMode = '',

    [ValidateSet('', 'ollama')]
    [string] $ExpectedRole = '',

    [ValidateSet('127.0.0.1', '0.0.0.0')]
    [string] $ExpectedHostIp = '127.0.0.1',

    [switch] $AppOnly,
    [switch] $NoPublishedPorts,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ExpectedWorkingDir
)

$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 otherwise decodes redirected native output with the
# active legacy code page, which corrupts Compose working_dir labels containing
# Unicode. Docker emits UTF-8 JSON.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8NoBom

function Write-LaunchInspection {
    param(
        [Parameter(Mandatory = $true)]
        [string] $State,
        [string] $AppPort = '',
        [string] $ComfyPort = '',
        [string] $Message = ''
    )

    # These lines are parsed by cmd.exe. Strip shell metacharacters even though
    # all current messages are fixed strings.
    $safeMessage = $Message -replace '[\r\n|%&<>^()!]', ' '
    [Console]::Out.WriteLine(('STATE={0}' -f $State))
    [Console]::Out.WriteLine(('APP_PORT={0}' -f $AppPort))
    [Console]::Out.WriteLine(('COMFY_PORT={0}' -f $ComfyPort))
    [Console]::Out.WriteLine(('MESSAGE={0}' -f $safeMessage))
}

function Normalize-DirectoryPath {
    param([string] $Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    try {
        $fullPath = [System.IO.Path]::GetFullPath($Value)
    }
    catch {
        return $null
    }

    $normalized = $fullPath.Replace('/', '\')
    while ($normalized.Length -gt 3 -and $normalized.EndsWith('\')) {
        $normalized = $normalized.Substring(0, $normalized.Length - 1)
    }
    return $normalized.ToUpperInvariant()
}

function Get-PublishedPort {
    param(
        [AllowNull()]
        [object] $Ports,
        [Parameter(Mandatory = $true)]
        [string] $ContainerPort,
        [Parameter(Mandatory = $true)]
        [bool] $Required,
        [Parameter(Mandatory = $true)]
        [ValidateSet('127.0.0.1', '0.0.0.0')]
        [string] $ExpectedHostIp,
        [Parameter(Mandatory = $true)]
        [ref] $BindingMismatch
    )

    if ($null -eq $Ports) {
        if ($Required) {
            throw "Missing published ports."
        }
        return $null
    }

    $property = $Ports.PSObject.Properties[$ContainerPort]
    if ($null -eq $property -or $null -eq $property.Value) {
        if ($Required) {
            throw "Missing published port."
        }
        return $null
    }

    $allowedHostIps = if ($ExpectedHostIp -eq '127.0.0.1') {
        @('127.0.0.1', '::1')
    }
    else {
        @('0.0.0.0', '::')
    }

    $published = @()
    foreach ($binding in @($property.Value)) {
        $parsedPort = 0
        if ($null -eq $binding -or
            -not [int]::TryParse([string] $binding.HostPort, [ref] $parsedPort) -or
            $parsedPort -lt 1 -or $parsedPort -gt 65535) {
            throw "Invalid published port."
        }
        $actualHostIp = [string] $binding.HostIp
        if ($actualHostIp -notin $allowedHostIps) {
            $BindingMismatch.Value = $true
        }
        $published += $parsedPort
    }

    $unique = @($published | Sort-Object -Unique)
    if ($unique.Count -ne 1) {
        throw "Missing or ambiguous published port."
    }
    return [int] $unique[0]
}

try {
    # Windows PowerShell 5.1 turns native stderr into ErrorRecord objects. Keep
    # those records as inspect text instead of letting a normal "not found"
    # response become a terminating PowerShell error.
    $ErrorActionPreference = 'Continue'
    $inspectOutput = @(& $DockerExe inspect --type container $ContainerName 2>&1)
    $inspectExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $inspectText = ($inspectOutput | ForEach-Object { [string] $_ }) -join [Environment]::NewLine

    if ($inspectExitCode -ne 0) {
        if ($inspectText -match '(?i)No such (object|container)') {
            Write-LaunchInspection -State 'ABSENT'
        }
        else {
            Write-LaunchInspection -State 'INVALID' -Message 'Docker could not safely inspect the existing container.'
        }
        exit 0
    }

    $decoded = ConvertFrom-Json -InputObject $inspectText
    $containers = @($decoded)
    if ($containers.Count -ne 1) {
        Write-LaunchInspection -State 'INVALID' -Message 'Docker returned an unexpected inspection result.'
        exit 0
    }

    $container = $containers[0]
    $labels = $container.Config.Labels
    $actualProject = [string] $labels.'com.docker.compose.project'
    $actualService = [string] $labels.'com.docker.compose.service'
    $actualWorkingDir = Normalize-DirectoryPath (
        [string] $labels.'com.docker.compose.project.working_dir'
    )
    $wantedWorkingDir = Normalize-DirectoryPath $ExpectedWorkingDir

    if ($actualProject -cne $ExpectedProject -or
        $actualService -cne $ExpectedService -or
        $null -eq $actualWorkingDir -or
        $null -eq $wantedWorkingDir -or
        $actualWorkingDir -cne $wantedWorkingDir) {
        Write-LaunchInspection -State 'COLLISION' -Message 'The container name belongs to another Docker project or folder.'
        exit 0
    }
    if ($ExpectedRole) {
        $actualRole = [string] $labels.'io.lora-dataset-studio.role'
        if ($actualRole -cne $ExpectedRole) {
            Write-LaunchInspection -State 'COLLISION' -Message 'The container has no matching LDS ownership role.'
            exit 0
        }
    }
    $needsSwitch = $false
    $actualMode = [string] $labels.'io.lora-dataset-studio.launch-mode'
    if ($ExpectedMode) {
        # Legacy containers are recognizable only under the two fixed projects.
        if (-not $actualMode) {
            if ($actualProject -ceq 'lora-dataset-studio-gpu') {
                $actualMode = 'gpu'
            }
            elseif ($actualProject -ceq 'lora-dataset-studio') {
                $actualMode = 'studio'
            }
        }
        if (-not $actualMode) {
            Write-LaunchInspection -State 'COLLISION' -Message 'The managed container has no verifiable launch mode.'
            exit 0
        }
        if ($actualMode -cne $ExpectedMode) {
            $needsSwitch = $true
        }
    }

    if ($ExpectedOllamaMode) {
        $actualOllamaMode = [string] $labels.'io.lora-dataset-studio.ollama-mode'
        if (-not $actualOllamaMode) {
            $actualOllamaMode = 'none'
        }
        if ($actualOllamaMode -cne $ExpectedOllamaMode) {
            $needsSwitch = $true
        }
    }


    $status = ([string] $container.State.Status).ToLowerInvariant()
    if ($status -ne 'running' -and
        $status -ne 'created' -and
        $status -ne 'exited' -and
        $status -ne 'dead') {
        Write-LaunchInspection -State 'INVALID' -Message 'The existing container is in a transitional or unsupported state.'
        exit 0
    }

    $appPort = $null
    $comfyPort = $null
    if (-not $NoPublishedPorts) {
        $bindingMismatch = $false
        $ports = $container.NetworkSettings.Ports
        $portsRequired = $status -eq 'running'
        try {
            $appPort = Get-PublishedPort `
                -Ports $ports `
                -ContainerPort '5050/tcp' `
                -Required $portsRequired `
                -ExpectedHostIp $ExpectedHostIp `
                -BindingMismatch ([ref] $bindingMismatch)
            $comfyRequired = $portsRequired -and -not $AppOnly -and -not $needsSwitch
            $comfyPort = Get-PublishedPort `
                -Ports $ports `
                -ContainerPort '8188/tcp' `
                -Required $comfyRequired `
                -ExpectedHostIp $ExpectedHostIp `
                -BindingMismatch ([ref] $bindingMismatch)
        }
        catch {
            Write-LaunchInspection -State 'INVALID' -Message 'The existing container has invalid or ambiguous published ports.'
            exit 0
        }
        if ($bindingMismatch) {
            $needsSwitch = $true
        }
    }

    if ($needsSwitch) {
        Write-LaunchInspection -State 'SWITCH' -AppPort $appPort -ComfyPort $comfyPort
    }
    elseif ($status -eq 'running') {
        Write-LaunchInspection -State 'RUNNING' -AppPort $appPort -ComfyPort $comfyPort
    }
    else {
        Write-LaunchInspection -State 'STOPPED' -AppPort $appPort -ComfyPort $comfyPort
    }
}
catch {
    Write-LaunchInspection -State 'INVALID' -Message 'Docker inspection failed before startup.'
}

exit 0
