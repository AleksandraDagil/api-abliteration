param(
    [double]$CostLimit = 120
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repository

$environmentFile = Join-Path $repository '.env'
$keyLine = Get-Content -LiteralPath $environmentFile |
    Where-Object { $_ -match '^\s*ABLITERATION_API_KEY_LARGE\s*=\s*\S+' } |
    Select-Object -First 1
if ($null -eq $keyLine) {
    throw 'No non-empty ABLITERATION_API_KEY_LARGE entry found in .env'
}
$keyValue = ($keyLine -split '=', 2)[1].Trim()
if (($keyValue.StartsWith("'") -and $keyValue.EndsWith("'")) -or
    ($keyValue.StartsWith('"') -and $keyValue.EndsWith('"'))) {
    $keyValue = $keyValue.Substring(1, $keyValue.Length - 2)
}

$runId = 'abliterated_model_large_bio_full'
$runDirectory = Join-Path $repository "outputs\api_runs\$runId"
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$pidPath = Join-Path $runDirectory 'process.pid'
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -Raw -LiteralPath $pidPath)
    $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $existingProcess) {
        throw "The large Bio run is already active as process $existingPid"
    }
}

$env:ABLITERATION_API_KEY_LARGE = $keyValue
$env:HF_HOME = Join-Path $repository '.cache\huggingface'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'

$python = Join-Path $repository '.venv\Scripts\python.exe'
$arguments = @(
    'api_main.py',
    'api_experiment=abliterated_model_large_bio',
    'provider.api_key_env=ABLITERATION_API_KEY_LARGE',
    'execution.live=true',
    'execution.resume=true',
    'execution.workers=4',
    "execution.max_total_estimated_cost_usd=$CostLimit",
    "output.run_id=$runId"
)
$stdoutPath = Join-Path $runDirectory 'process.stdout.log'
$stderrPath = Join-Path $runDirectory 'process.stderr.log'
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repository `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ASCII
Write-Output "Started large Bio run as process $($process.Id) with a `$$CostLimit ceiling"
Write-Output "Run directory: $runDirectory"
