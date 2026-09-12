$ErrorActionPreference = 'Stop'

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repository

$environmentFile = Join-Path $repository '.env'
$keyLine = Get-Content -LiteralPath $environmentFile |
    Where-Object { $_ -match '^\s*ABLIT_KEY\s*=\s*\S+' } |
    Select-Object -First 1
if ($null -eq $keyLine) {
    throw 'No non-empty ABLIT_KEY entry found in .env'
}
$keyValue = ($keyLine -split '=', 2)[1].Trim()
if (($keyValue.StartsWith("'") -and $keyValue.EndsWith("'")) -or
    ($keyValue.StartsWith('"') -and $keyValue.EndsWith('"'))) {
    $keyValue = $keyValue.Substring(1, $keyValue.Length - 2)
}

$runDirectory = Join-Path $repository 'outputs\api_runs\abliterated_model_bio_full'
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$pidPath = Join-Path $runDirectory 'process.pid'
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -Raw -LiteralPath $pidPath)
    $existingProcess = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $existingProcess) {
        throw "The full Bio run is already active as process $existingPid"
    }
}

$env:ABLIT_KEY = $keyValue
$env:HF_HOME = Join-Path $repository '.cache\huggingface'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'

$python = Join-Path $repository '.venv\Scripts\python.exe'
$arguments = @(
    'api_main.py',
    'api_experiment=abliterated_model_bio_full'
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
Write-Output "Started full Bio run as process $($process.Id)"
Write-Output "Run directory: $runDirectory"
