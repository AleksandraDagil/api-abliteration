param(
    [Parameter(Mandatory = $true)]
    [int]$CurrentProcessId
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runDirectory = Join-Path $repository 'outputs\api_runs\abliterated_model_bio_full'
$watcherLog = Join-Path $runDirectory 'ceiling_upgrade_watcher.log'

try {
    Wait-Process -Id $CurrentProcessId -ErrorAction Stop
} catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
    # The process ended between the status check and watcher startup.
}

Start-Sleep -Seconds 5
$timestamp = [datetimeoffset]::UtcNow.ToString('o')
Add-Content -LiteralPath $watcherLog -Value "$timestamp restarting with updated ceiling"
& (Join-Path $PSScriptRoot 'start_full_bio_background.ps1') |
    Add-Content -LiteralPath $watcherLog
