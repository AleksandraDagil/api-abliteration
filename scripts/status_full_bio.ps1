$ErrorActionPreference = 'Stop'

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runDirectory = Join-Path $repository 'outputs\api_runs\abliterated_model_bio_full'
$pidPath = Join-Path $runDirectory 'process.pid'
$manifestPath = Join-Path $runDirectory 'manifest.json'
$transcriptPath = Join-Path $runDirectory 'transcript.csv'

$processStatus = 'not started'
if (Test-Path -LiteralPath $pidPath) {
    $processId = [int](Get-Content -Raw -LiteralPath $pidPath)
    if ($null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
        $processStatus = "running (PID $processId)"
    } else {
        $processStatus = "not running (last PID $processId)"
    }
}

$manifestStatus = 'not created'
$completedRows = 0
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifestStatus = $manifest.status
    $completedRows = $manifest.completed_rows
}
if (Test-Path -LiteralPath $transcriptPath) {
    $completedRows = @(Import-Csv -LiteralPath $transcriptPath).Count
}

[pscustomobject]@{
    Process = $processStatus
    Manifest = $manifestStatus
    CompletedSamples = $completedRows
    TotalSamples = 1556
    Transcript = $transcriptPath
} | Format-List
