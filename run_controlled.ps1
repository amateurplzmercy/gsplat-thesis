param([switch]$KeepPowerPlan, [switch]$CompareLayouts)

$ErrorActionPreference = 'Stop'
$performance = '27fa6203-3987-4dcc-918d-748559d549ec'
$pattern = '[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}'
$original = [regex]::Match((powercfg /getactivescheme | Out-String), $pattern).Value
if (-not $original) { throw 'Could not identify the original power plan.' }
$previousBlas = $env:OPENBLAS_NUM_THREADS
$previousOmp = $env:OMP_NUM_THREADS
$code = 1
$switched = $false
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $KeepPowerPlan -and $original -ne $performance) {
        powercfg /setactive $performance
        if ($LASTEXITCODE -ne 0) { throw 'Could not activate the Performance plan.' }
        $switched = $true
    }
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $module = if ($CompareLayouts) { 'scripts.compare_block_layouts' } else { 'scripts.run_reliable_benchmark' }
    & "$PSScriptRoot\run_python.ps1" -m $module @args
    $code = $LASTEXITCODE
}
finally {
    $env:OPENBLAS_NUM_THREADS = $previousBlas
    $env:OMP_NUM_THREADS = $previousOmp
    if ($switched) {
        powercfg /setactive $original
        if ($LASTEXITCODE -ne 0) { Write-Error "Could not restore original plan $original." }
    }
    powercfg /getactivescheme
    Pop-Location
}
exit $code
