$ErrorActionPreference = "Continue"

# ============================================================
# MULTI-SEED FINAL RUN - RICE LEAF HYBRID CNN
# No borra runs actuales. Agrega nuevas ejecuciones.
# Usa split ya corregido por pHash y features ya generadas.
# ============================================================

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\multiseed" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\tables" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\figures" | Out-Null
New-Item -ItemType Directory -Force -Path "archive" | Out-Null
New-Item -ItemType Directory -Force -Path "runs" | Out-Null

$env:PYTHONFAULTHANDLER = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$startTime = Get-Date
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveDir = "archive\before_multiseed_$stamp"
$summary = @()

function Run-Step {
    param (
        [string]$Name,
        [string]$Command,
        [string]$LogFile,
        [bool]$Critical = $false
    )

    Write-Host ""
    Write-Host "========================================="
    Write-Host "STEP: $Name"
    Write-Host "========================================="
    Write-Host "Command: $Command"
    Write-Host ""

    $stepStart = Get-Date

    cmd /c $Command 2>&1 | Tee-Object -FilePath $LogFile | ForEach-Object {
        Write-Host $_
    }

    $exitCode = [int]$LASTEXITCODE
    $stepEnd = Get-Date
    $duration = New-TimeSpan -Start $stepStart -End $stepEnd

    if ($exitCode -eq 0) {
        Write-Host "OK: $Name finalizado correctamente en $($duration.ToString())"

        $script:summary += [PSCustomObject]@{
            Step = $Name
            Status = "OK"
            ExitCode = $exitCode
            Duration = $duration.ToString()
            Log = $LogFile
        }
    }
    else {
        Write-Host "ERROR: $Name falló con exit code $exitCode"

        $script:summary += [PSCustomObject]@{
            Step = $Name
            Status = "ERROR"
            ExitCode = $exitCode
            Duration = $duration.ToString()
            Log = $LogFile
        }

        if ($Critical) {
            Write-Host "Paso crítico fallido. Se detiene el pipeline."
            $script:summary | Export-Csv "logs\multiseed_summary.csv" -NoTypeInformation -Encoding UTF8
            exit $exitCode
        }
    }

    $global:LAST_STEP_EXIT_CODE = $exitCode
}

function Set-Project-Seed {
    param (
        [int]$Seed
    )

    Write-Host ""
    Write-Host "Ajustando project.seed a $Seed..."

    python -c "from pathlib import Path; import re; p=Path('configs/default.yml'); s=p.read_text(encoding='utf-8'); s=re.sub(r'(?m)^(\s*seed:\s*)\d+', r'\g<1>$Seed', s, count=1); p.write_text(s, encoding='utf-8'); print('default.yml project.seed actualizado a $Seed')"
}

function Set-BatchSize-8 {
    Write-Host ""
    Write-Host "Asegurando batch_size=8 en default.yml y E3..."

    python -c "from pathlib import Path; import re; files=[Path('configs/default.yml'), Path('configs/experiments/E3_proposed_segmented_cnn_texture.yml')]; [p.write_text(re.sub(r'batch_size:\s*\d+', 'batch_size: 8', p.read_text(encoding='utf-8')), encoding='utf-8') for p in files]; print('batch_size=8 aplicado')"
}

function Validate-Experiment-Configs {
    Write-Host ""
    Write-Host "Validando configs E0-E3..."

    python -c "from riceleaf.config.loader import load_config; files=['E0_baseline_cnn.yml','E1_segmented_cnn.yml','E2_cnn_texture.yml','E3_proposed_segmented_cnn_texture.yml']; [print(f, 'seed=', load_config('configs/experiments/'+f).project.seed, 'batch_size=', load_config('configs/experiments/'+f).training.batch_size, 'epochs=', load_config('configs/experiments/'+f).training.epochs) for f in files]"
}

Write-Host "========================================="
Write-Host "INICIO MULTI-SEED FINAL RUN"
Write-Host "Inicio: $startTime"
Write-Host "Archivo de respaldo: $archiveDir"
Write-Host "========================================="

# ============================================================
# 00. Backup before doing anything
# ============================================================

New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null

if (Test-Path "runs") {
    Copy-Item -Recurse -Force "runs" "$archiveDir\runs"
}

if (Test-Path "outputs") {
    Copy-Item -Recurse -Force "outputs" "$archiveDir\outputs"
}

if (Test-Path "logs") {
    Copy-Item -Recurse -Force "logs" "$archiveDir\logs"
}

Write-Host "OK: respaldo creado en $archiveDir"

# ============================================================
# 01. Validate environment and data
# ============================================================

Run-Step `
    -Name "00 Check environment" `
    -Command "python scripts/00_check_environment.py" `
    -LogFile "logs\multiseed\00_check_environment.log" `
    -Critical $true

Run-Step `
    -Name "01 Add split metadata columns" `
    -Command "python scripts/01c_add_split_metadata.py" `
    -LogFile "logs\multiseed\01c_add_split_metadata.log" `
    -Critical $true

Run-Step `
    -Name "02 Validate split schema" `
    -Command "python -c ""import pandas as pd; required={'image_id','relative_path','label','label_idx'}; total=0; missing=[]; 
for s in ['train','val','test']:
    df=pd.read_csv(f'data/splits/{s}.csv')
    print(s, len(df), df.columns.tolist())
    total += len(df)
    missing.extend([(s,c) for c in required-set(df.columns)])
assert not missing, f'Missing columns: {missing}'
assert total==5932, f'Total esperado 5932, encontrado {total}'
print('OK split schema y total:', total)""" `
    -LogFile "logs\multiseed\02_validate_split_schema.log" `
    -Critical $true

if (Test-Path "check_near_duplicates.py") {
    Run-Step `
        -Name "03 Validate no visual duplicates between splits" `
        -Command "python check_near_duplicates.py" `
        -LogFile "logs\multiseed\03_check_near_duplicates.log" `
        -Critical $true
}

Run-Step `
    -Name "04 Validate feature files exist" `
    -Command "python -c ""from pathlib import Path; required=['data/features/glcm_dwt_original/train_features_scaled.csv','data/features/glcm_dwt_original/val_features_scaled.csv','data/features/glcm_dwt_original/test_features_scaled.csv','data/features/glcm_dwt_segmented/train_features_scaled.csv','data/features/glcm_dwt_segmented/val_features_scaled.csv','data/features/glcm_dwt_segmented/test_features_scaled.csv']; missing=[p for p in required if not Path(p).exists()]; print('missing:', missing); assert not missing, f'Faltan features: {missing}'; print('OK features listas')""" `
    -LogFile "logs\multiseed\04_validate_features.log" `
    -Critical $true

# ============================================================
# 02. Configs
# ============================================================

Set-BatchSize-8
Validate-Experiment-Configs

# ============================================================
# 03. Run additional seeds
# Seed 42 ya existe. Agregamos cuatro seeds para total n=5.
# ============================================================

$Seeds = @(123, 2026, 7, 99)

foreach ($Seed in $Seeds) {
    Write-Host ""
    Write-Host "#################################################"
    Write-Host "INICIO SEED $Seed"
    Write-Host "#################################################"

    Set-Project-Seed -Seed $Seed
    Validate-Experiment-Configs

    Run-Step `
        -Name "Train E0 baseline CNN seed $Seed" `
        -Command "python scripts/04_train_experiment.py --config configs/experiments/E0_baseline_cnn.yml" `
        -LogFile "logs\multiseed\train_E0_seed_$Seed.log" `
        -Critical $true

    Run-Step `
        -Name "Train E1 segmented CNN seed $Seed" `
        -Command "python scripts/04_train_experiment.py --config configs/experiments/E1_segmented_cnn.yml" `
        -LogFile "logs\multiseed\train_E1_seed_$Seed.log" `
        -Critical $true

    Run-Step `
        -Name "Train E2 CNN texture seed $Seed" `
        -Command "python scripts/04_train_experiment.py --config configs/experiments/E2_cnn_texture.yml" `
        -LogFile "logs\multiseed\train_E2_seed_$Seed.log" `
        -Critical $true

    Run-Step `
        -Name "Train E3 proposed segmented CNN texture seed $Seed" `
        -Command "python scripts/04_train_experiment.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml" `
        -LogFile "logs\multiseed\train_E3_seed_$Seed.log" `
        -Critical $true

    Write-Host ""
    Write-Host "FIN SEED $Seed"
}

# Restaurar seed principal a 42
Set-Project-Seed -Seed 42
Set-BatchSize-8
Validate-Experiment-Configs

# ============================================================
# 04. Compare, export, aggregate
# ============================================================

Run-Step `
    -Name "Compare all experiments" `
    -Command "python scripts/06_compare_experiments.py --runs runs/" `
    -LogFile "logs\multiseed\compare_all_experiments.log" `
    -Critical $true

Run-Step `
    -Name "Export paper assets" `
    -Command "python scripts/07_export_paper_assets.py" `
    -LogFile "logs\multiseed\export_paper_assets.log" `
    -Critical $true

Run-Step `
    -Name "Aggregate multiseed results" `
    -Command "python scripts/08_aggregate_multiseed_results.py" `
    -LogFile "logs\multiseed\aggregate_multiseed_results.log" `
    -Critical $true

Run-Step `
    -Name "Print multiseed summary" `
    -Command "python -c ""import pandas as pd; print(pd.read_csv('outputs/tables/multiseed_summary_mean_std.csv').to_string())""" `
    -LogFile "logs\multiseed\print_multiseed_summary.log" `
    -Critical $true

Run-Step `
    -Name "Print paired deltas vs E0" `
    -Command "python -c ""import pandas as pd; print(pd.read_csv('outputs/tables/paired_deltas_vs_E0.csv').to_string())""" `
    -LogFile "logs\multiseed\print_paired_deltas.log" `
    -Critical $false

$endTime = Get-Date
$totalDuration = New-TimeSpan -Start $startTime -End $endTime

$summary | Export-Csv "logs\multiseed_summary.csv" -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "========================================="
Write-Host "FIN MULTI-SEED FINAL RUN"
Write-Host "Inicio: $startTime"
Write-Host "Fin: $endTime"
Write-Host "Duración total: $($totalDuration.ToString())"
Write-Host "Resumen guardado en logs\multiseed_summary.csv"
Write-Host "Backup inicial en $archiveDir"
Write-Host "========================================="
