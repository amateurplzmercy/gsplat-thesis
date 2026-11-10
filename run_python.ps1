# All arguments are passed to the project's isolated Python, without a global activation.
$runtime = Join-Path $PSScriptRoot 'tools\python310\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    throw "Project Python is missing: $runtime"
}
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousCache = $env:PIP_CACHE_DIR
try {
    $env:TEMP = Join-Path $PSScriptRoot 'tools\tmp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $PSScriptRoot 'downloads\pip_cache'
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    & $runtime -X utf8 -s @args
    $pythonExitCode = $LASTEXITCODE
}
finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:PIP_CACHE_DIR = $previousCache
}
exit $pythonExitCode
