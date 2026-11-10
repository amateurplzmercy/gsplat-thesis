param([string]$OutputRoot = 'reports')

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & "$PSScriptRoot\run_python.ps1" -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    & "$PSScriptRoot\run_python.ps1" scripts/check_environment.py --probe-torch --output (Join-Path $OutputRoot 'environment_cuda.json')
    if ($LASTEXITCODE -ne 0) { throw 'Environment inspection failed.' }
    & "$PSScriptRoot\run_python.ps1" scripts/validate_cuda.py --output (Join-Path $OutputRoot 'cuda_primitives.json')
    if ($LASTEXITCODE -ne 0) { throw 'CUDA primitive check failed.' }
    & "$PSScriptRoot\run_python.ps1" scripts/smoke_render.py --output-dir (Join-Path $OutputRoot 'smoke')
    if ($LASTEXITCODE -ne 0) { throw 'gsplat render check failed.' }
}
finally {
    Pop-Location
}
