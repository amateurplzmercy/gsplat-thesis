param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('env', 'data', 'main', 'analyze', 'demo')]
    [string]$Stage,
    [string]$OutputRoot,
    [switch]$Download,
    [switch]$KeepPowerPlan,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked([string]$Program, [string[]]$CommandArgs) {
    & $Program @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Assert-FileHash([string]$Path, [string]$Sha256) {
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA256 mismatch for $Path. Inspect the cached file before retrying."
    }
}

function Prepare-Environment {
    $runtime = 'tools/python310/python.exe'
    if (-not (Test-Path -LiteralPath $runtime)) {
        $bootstrap = @('-3', 'scripts/prepare_runtime.py', '--runtime-only', '--sha256', $sources.python_archive.sha256)
        if (-not $Download) { $bootstrap += '--offline' }
        Invoke-Checked 'py' $bootstrap
    }
    $version = & '.\run_python.ps1' --version
    if ($LASTEXITCODE -ne 0 -or $version -ne 'Python 3.10.11') {
        throw "Expected project-local Python 3.10.11, found $version"
    }
    $links = @('--only-binary=:all:', '--require-hashes', '--find-links', 'downloads', '--find-links', 'downloads/wheels')
    New-Item -ItemType Directory -Force -Path 'downloads/wheels', 'tools/tmp' | Out-Null
    $index = if ($Download) { @('--index-url', 'https://pypi.org/simple') } else { @('--no-index') }
    if (-not (Test-Path -LiteralPath 'tools/python310/Lib/site-packages/pip/__main__.py')) {
        $python = (Resolve-Path -LiteralPath $runtime).Path
        Invoke-Checked 'py' (@('-3', '-m', 'pip', '--isolated', '--python', $python, 'install', '--no-cache-dir') +
            $links + $index + @('-r', 'configs/bootstrap-pip.txt'))
    }
    if ($Download) {
        $packages = & '.\run_python.ps1' -m pip list --format=json
        if ($LASTEXITCODE -ne 0) { throw 'Could not list installed packages.' }
        $installed = @{}
        foreach ($package in ($packages | ConvertFrom-Json)) { $installed[$package.name] = $package.version }
        foreach ($wheel in $sources.cuda_wheels) {
            $name, $version = ($wheel.filename -split '-', 3)[0..1]
            $target = Join-Path 'downloads' $wheel.filename
            if ($installed[$name] -ne $version -and -not (Test-Path -LiteralPath $target)) {
                Invoke-Checked '.\run_python.ps1' @('scripts/download_wheel.py', $wheel.url, $target, '--sha256', $wheel.sha256)
            }
        }
    }
    Invoke-Checked '.\run_python.ps1' (@('-m', 'pip', '--isolated', '--disable-pip-version-check', 'install') +
        $links + $index + @('-r', 'requirements-lock.txt'))
    & '.\verify_environment.ps1' -OutputRoot (Join-Path $root 'env')
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
}

function Prepare-Scene([string]$Name, $Spec) {
    $folder = Join-Path 'data' $Name
    $missing = @($Spec.files.PSObject.Properties | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $folder $_.Name) -PathType Leaf)
    })
    if ($missing.Count) {
        if (-not $Download) { throw "Missing files in $folder. Supply the files listed in MANIFEST.md or rerun data with -Download." }
        Invoke-Checked '.\run_python.ps1' @('scripts/fetch_graphdeco_scene.py', '--scene', $Name, '--max-mib', $Spec.max_mib)
    }
    $sourcePath = Join-Path $folder 'source.json'
    if (Test-Path -LiteralPath $sourcePath) {
        $source = Get-Content -Raw -Encoding UTF8 -LiteralPath $sourcePath | ConvertFrom-Json
    }
    else {
        $source = [pscustomobject]@{source_url = $sources.model_archive; scene = $Name; iteration = 30000; files = $Spec.files}
    }
    if ($source.source_url -ne $sources.model_archive -or $source.scene -ne $Name -or $source.iteration -ne 30000) {
        throw "Unexpected scene metadata in $sourcePath"
    }
    foreach ($file in $Spec.files.PSObject.Properties) {
        Assert-FileHash (Join-Path $folder $file.Name) $file.Value.sha256
        $record = $source.files.PSObject.Properties[$file.Name].Value
        if ($record.sha256 -ne $file.Value.sha256 -or $record.archive_member -ne $file.Value.archive_member) {
            throw "Unexpected file metadata for $($file.Name) in $sourcePath"
        }
    }
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        [IO.File]::WriteAllText((Join-Path $PSScriptRoot $sourcePath), ($source | ConvertTo-Json -Depth 8))
    }
    Write-Host "Verified scene $Name"
}

function Prepare-Layout([string]$Scene, [int]$LeafSize) {
    $blocks = "$Scene/blocks-$LeafSize-banded"
    $indexPath = Join-Path $blocks 'index.json'
    if ((Test-Path -LiteralPath $indexPath) -and
        (Test-Path -LiteralPath (Join-Path $blocks 'payload.npy')) -and
        (Test-Path -LiteralPath (Join-Path $blocks 'original_ids.npy'))) {
        $index = Get-Content -Raw -Encoding UTF8 -LiteralPath $indexPath | ConvertFrom-Json
        $source = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $Scene 'source.json') | ConvertFrom-Json
        if ($index.leaf_size -eq $LeafSize -and
            ($index.scale_bands -join ',') -eq (@(0.04, 0.16, 0.64) -join ',') -and
            ($index.source | ConvertTo-Json -Compress -Depth 8) -eq ($source | ConvertTo-Json -Compress -Depth 8)) {
            return
        }
    }
    Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.spatial_blocks', '--scene', $Scene, '--output', $blocks,
        '--leaf-size', $LeafSize, '--scale-bands', '0.04', '0.16', '0.64')
}

function Invoke-Experiment($Job) {
    $options = @{}
    if ($KeepPowerPlan) { $options.KeepPowerPlan = $true }
    if ($Job.Name -eq 'layout-pair') { $options.CompareLayouts = $true }
    $command = $Job.Command + @('--output', (Join-Path $root $Job.Name))
    & '.\run_controlled.ps1' @options @command
    if ($LASTEXITCODE -ne 0) { throw "Experiment $($Job.Name) failed with exit code $LASTEXITCODE" }
}

Push-Location -LiteralPath $PSScriptRoot
$oldBlas = $env:OPENBLAS_NUM_THREADS
$oldOmp = $env:OMP_NUM_THREADS
try {
    if (-not $OutputRoot) { $OutputRoot = if ($Quick) { 'reports/reproduction-quick' } else { 'reports/reproduction' } }
    $root = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
    $sources = Get-Content -Raw -Encoding UTF8 configs/reproduction_sources.json | ConvertFrom-Json
    $playroom = 'data/playroom/blocks-2048-banded'
    $fine = 'data/playroom/blocks-512-banded'
    $train = 'data/train/blocks-2048-banded'
    $playroomWork = Join-Path $root 'workloads/playroom'
    $trainWork = Join-Path $root 'workloads/train'
    $capture = Join-Path $playroomWork 'capture_order.json'
    $matrix = Join-Path $root 'candidate-matrix'
    $mainComplete = Join-Path $root 'main-complete.json'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'

    $jobs = @(
        @{Name = 'candidate-matrix'; Command = @('--candidate-matrix', '--rounds', '16')}
        @{Name = 'train-221'; Command = @('--candidate-matrix', '--tracks', 'continuous', '--budget', '221',
                '--blocks', $train, '--workloads', $trainWork, '--rounds', '16')}
        @{Name = 'diagnostic'; Command = @('--diagnose', '--rounds', '4')}
        @{Name = 'diagnostic-capture'; Command = @('--diagnose', '--tracks', 'capture_order', '--rounds', '4')}
        @{Name = 'formal'; Command = @('--rounds', '16')}
        @{Name = 'ablation-pipeline'; Command = @('--ablation', 'pipeline', '--budget', '326', '--rounds', '16')}
        @{Name = 'ablation-admission'; Command = @('--ablation', 'admission', '--budget', '326', '--rounds', '16')}
        @{Name = 'candidate'; Command = @('--ablation', 'candidate', '--budget', '326', '--rounds', '16')}
        @{Name = 'layout-pair'; Command = @('--coarse', $playroom, '--fine', $fine, '--trajectory', $capture,
                '--budget', '375', '--rounds', '16')}
        @{Name = 'short-pipeline'; Command = @('--ablation', 'short_pipeline', '--budget', '326', '--rounds', '16')}
        @{Name = 'short-pipeline-stress'; Command = @('--ablation', 'short_pipeline', '--tracks', 'return', 'capture_order', '--rounds', '16')}
        @{Name = 'short-pipeline-confirm'; Command = @('--ablation', 'short_pipeline_confirm', '--tracks', 'capture_order',
                '--budget', '375', '--rounds', '16')}
    )
    if ($Quick) {
        $jobs = @(@{Name = 'candidate-matrix'; Command = @('--candidate-matrix', '--tracks', 'continuous', 'capture_order', '--rounds', '4')})
    }
    foreach ($job in $jobs) {
        if ($job.Name -notin @('train-221', 'layout-pair')) { $job.Command += @('--workloads', $playroomWork) }
    }

    switch ($Stage) {
        'env' { Prepare-Environment }
        'data' {
            Prepare-Scene 'train' $sources.scenes.train
            Prepare-Scene 'playroom' $sources.scenes.playroom
            Prepare-Layout 'data/train' 2048
            Prepare-Layout 'data/playroom' 2048
            if (-not $Quick) { Prepare-Layout 'data/playroom' 512 }
            Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.prepare_workloads', '--scene', 'data/train',
                '--blocks', $train, '--output', $trainWork)
            Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.prepare_workloads', '--scene', 'data/playroom',
                '--blocks', $playroom, '--output', $playroomWork)
        }
        'main' {
            if (Test-Path -LiteralPath $mainComplete) {
                throw "This output root already holds completed measurements. Choose a new -OutputRoot to remeasure."
            }
            if (-not (Test-Path -LiteralPath $capture)) { throw 'Run the data stage with this output root first.' }
            if (-not $Quick) {
                Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.audit_policy_rgb', '--workloads', $playroomWork,
                    '--output', (Join-Path $matrix 'rgb'))
                Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.audit_policy_rgb', '--blocks', $train,
                    '--workloads', $trainWork, '--tracks', 'continuous', '--budget', '221', '--output', (Join-Path $root 'train-221/rgb'))
                Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.stream_blocks', '--policy', 'audit', '--scene', 'data/playroom',
                    '--blocks', $fine, '--trajectory', $capture, '--budget-mib', '375', '--output', (Join-Path $root 'playroom-fine-rgb'))
                foreach ($check in @(
                    @{Track = 'continuous'; Budget = '326'; Name = 'short-pipeline'}
                    @{Track = 'return'; Budget = '298'; Name = 'short-pipeline-stress'}
                    @{Track = 'capture_order'; Budget = '375'; Name = 'short-pipeline-stress'}
                )) {
                    Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.audit_policy_rgb', '--short-pipeline', '--workloads', $playroomWork,
                        '--tracks', $check.Track, '--budget', $check.Budget, '--output', (Join-Path $root "$($check.Name)/rgb"))
                }
            }
            foreach ($job in $jobs) { Invoke-Experiment $job }
            @{quick = [bool]$Quick; experiments = @($jobs.Name)} | ConvertTo-Json |
                Set-Content -LiteralPath $mainComplete -Encoding UTF8
        }
        'analyze' {
            if (-not (Test-Path -LiteralPath $mainComplete)) { throw 'Complete the main stage for this output root first.' }
            $completed = Get-Content -Raw -Encoding UTF8 -LiteralPath $mainComplete | ConvertFrom-Json
            if ($completed.quick -ne [bool]$Quick) { throw 'Use the same -Quick setting as the main stage.' }
            foreach ($name in $completed.experiments) {
                Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.analyze_paired', (Join-Path $root $name))
            }
            if (-not $Quick) {
                Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.diagnose_working_set', '--scene', 'data/playroom', '--blocks', $playroom,
                    '--trajectory', $capture, '--budget', '375', '--timings', (Join-Path $matrix 'capture_order/frames.csv'),
                    '--output', (Join-Path $root 'playroom-capture'))
            }
        }
        'demo' {
            if (-not (Test-Path -LiteralPath (Join-Path $matrix 'capture_order/paired_statistics.json'))) {
                throw 'Run analyze for this output root before exporting the demo.'
            }
            $demo = Join-Path $root 'demo'
            Invoke-Checked '.\run_python.ps1' @('-m', 'scripts.export_replay_demo', '--output', $demo,
                '--measurements', $matrix, '--workloads', $playroomWork)
            Copy-Item -LiteralPath 'demo/index.html', 'demo/replay.css', 'demo/replay.js' -Destination $demo
            Write-Host "Open $(Join-Path $demo 'index.html') in a browser."
        }
    }
    Write-Host "$Stage complete. Output root: $root"
}
finally {
    $env:OPENBLAS_NUM_THREADS = $oldBlas
    $env:OMP_NUM_THREADS = $oldOmp
    Pop-Location
}
