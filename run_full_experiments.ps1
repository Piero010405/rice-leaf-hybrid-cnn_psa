$ErrorActionPreference = "Continue"

# ============================================================
# PIPELINE FULL - RICE LEAF HYBRID CNN
# Split corregido por pHash, sin duplicados visuales entre splits
# Ejecución desde cero
# ============================================================

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\tables" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\figures" | Out-Null
New-Item -ItemType Directory -Force -Path "runs" | Out-Null

$env:PYTHONFAULTHANDLER = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$startTime = Get-Date
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

    cmd /c $Command 2>&1 | Tee-Object $LogFile

    $exitCode = $LASTEXITCODE
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
            $script:summary | Export-Csv "logs\pipeline_summary.csv" -NoTypeInformation -Encoding UTF8
            exit $exitCode
        }
    }

    return $exitCode
}

function Set-E3-BatchSize {
    param (
        [int]$BatchSize
    )

    Write-Host ""
    Write-Host "Ajustando batch_size de E3 a $BatchSize..."

    python -c "from pathlib import Path; import re; p=Path('configs/experiments/E3_proposed_segmented_cnn_texture.yml'); s=p.read_text(encoding='utf-8'); s=re.sub(r'batch_size:\s*\d+', 'batch_size: $BatchSize', s); p.write_text(s, encoding='utf-8'); print('E3 batch_size actualizado a $BatchSize')"
}

function Validate-E3-Config {
    Write-Host ""
    Write-Host "Validando config E3..."
    python -c "from riceleaf.config.loader import load_config; cfg=load_config('configs/experiments/E3_proposed_segmented_cnn_texture.yml'); print('experiment:', cfg.experiment.id, cfg.experiment.name); print('batch_size:', cfg.training.batch_size); print('epochs:', cfg.training.epochs)"
}

Write-Host "========================================="
Write-Host "INICIO PIPELINE FULL"
Write-Host "Inicio: $startTime"
Write-Host "========================================="

# ============================================================
# 00. Environment
# ============================================================

$null = Run-Step `
    -Name "00 Check environment" `
    -Command "python scripts/00_check_environment.py" `
    -LogFile "logs\00_check_environment.log" `
    -Critical $true

# ============================================================
# 01. Add/fix split metadata
# ============================================================

$null = Run-Step `
    -Name "01C Add split metadata columns" `
    -Command "python scripts/01c_add_split_metadata.py" `
    -LogFile "logs\01c_add_split_metadata.log" `
    -Critical $true

# ============================================================
# 01B. Validate split columns and counts
# ============================================================

$null = Run-Step `
    -Name "01D Validate split schema" `
    -Command "python -c ""import pandas as pd; required={'image_id','relative_path','label','label_idx'}; total=0; [print(s, len(pd.read_csv(f'data/splits/{s}.csv')), pd.read_csv(f'data/splits/{s}.csv').columns.tolist()) or globals().__setitem__('total', total+len(pd.read_csv(f'data/splits/{s}.csv'))) for s in ['train','val','test']]; missing=[]; [missing.extend([(s,c) for c in required-set(pd.read_csv(f'data/splits/{s}.csv').columns)]) for s in ['train','val','test']]; assert not missing, f'Missing columns: {missing}'; assert total==5932, f'Total esperado 5932, encontrado {total}'; print('OK split schema y total:', total)""" `
    -LogFile "logs\01d_validate_split_schema.log" `
    -Critical $true

# ============================================================
# 01C. Optional pHash leakage validation
# ============================================================

if (Test-Path "check_near_duplicates.py") {
    $null = Run-Step `
        -Name "01 Validate no visual duplicates between splits" `
        -Command "python check_near_duplicates.py" `
        -LogFile "logs\01_check_near_duplicates.log" `
        -Critical $true
}
else {
    Write-Host ""
    Write-Host "WARN: check_near_duplicates.py no existe. Se omite validación automática de duplicados visuales."
}

# ============================================================
# 01D. Clean generated artifacts for full rerun
# ============================================================

Write-Host ""
Write-Host "========================================="
Write-Host "LIMPIEZA DE ARTEFACTOS GENERADOS"
Write-Host "========================================="

Remove-Item -Recurse -Force "data\segmented" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "data\features" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "runs" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "outputs\tables" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "outputs\reports" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "outputs\figures" -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path "data\segmented" | Out-Null
New-Item -ItemType Directory -Force -Path "data\features" | Out-Null
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\tables" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\figures" | Out-Null

Write-Host "OK: limpieza completada."

# ============================================================
# 02. Generate segmented dataset
# ============================================================

$null = Run-Step `
    -Name "02 Generate segmented dataset" `
    -Command "python scripts/02_generate_segmented_dataset.py --config configs/experiments/E1_segmented_cnn.yml" `
    -LogFile "logs\02_generate_segmented_dataset.log" `
    -Critical $true

# ============================================================
# 03. Extract features
# ============================================================

$null = Run-Step `
    -Name "03 Extract features E2 original" `
    -Command "python scripts/03_extract_features.py --config configs/experiments/E2_cnn_texture.yml" `
    -LogFile "logs\03_extract_features_E2_original.log" `
    -Critical $true

$null = Run-Step `
    -Name "03 Extract features E3 segmented" `
    -Command "python scripts/03_extract_features.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml" `
    -LogFile "logs\03_extract_features_E3_segmented.log" `
    -Critical $true

# ============================================================
# 04. Train E0, E1, E2
# ============================================================

$null = Run-Step `
    -Name "04 Train E0 baseline CNN" `
    -Command "python scripts/04_train_experiment.py --config configs/experiments/E0_baseline_cnn.yml" `
    -LogFile "logs\04_train_E0_baseline_cnn.log" `
    -Critical $true

$null = Run-Step `
    -Name "04 Train E1 segmented CNN" `
    -Command "python scripts/04_train_experiment.py --config configs/experiments/E1_segmented_cnn.yml" `
    -LogFile "logs\04_train_E1_segmented_cnn.log" `
    -Critical $true

$null = Run-Step `
    -Name "04 Train E2 CNN texture" `
    -Command "python scripts/04_train_experiment.py --config configs/experiments/E2_cnn_texture.yml" `
    -LogFile "logs\04_train_E2_cnn_texture.log" `
    -Critical $true

# ============================================================
# 04B. Train E3 with retry mechanism
# ============================================================

Set-E3-BatchSize -BatchSize 8
Validate-E3-Config

Remove-Item -Recurse -Force "runs\E3_proposed_segmented_cnn_texture" -ErrorAction SilentlyContinue

$e3Exit = Run-Step `
    -Name "04 Train E3 proposed segmented CNN texture batch 8" `
    -Command "python scripts/04_train_experiment.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml" `
    -LogFile "logs\04_train_E3_proposed_batch8.log" `
    -Critical $false

if ($e3Exit -ne 0) {
    Write-Host ""
    Write-Host "E3 falló con batch_size 8. Se reintentará con batch_size 4."

    Set-E3-BatchSize -BatchSize 4
    Validate-E3-Config

    Remove-Item -Recurse -Force "runs\E3_proposed_segmented_cnn_texture" -ErrorAction SilentlyContinue

    $e3RetryExit = Run-Step `
        -Name "04 Train E3 proposed segmented CNN texture retry batch 4" `
        -Command "python scripts/04_train_experiment.py --config configs/experiments/E3_proposed_segmented_cnn_texture.yml" `
        -LogFile "logs\04_train_E3_proposed_batch4_retry.log" `
        -Critical $false

    if ($e3RetryExit -ne 0) {
        Write-Host ""
        Write-Host "ERROR: E3 falló incluso con batch_size 4. Se detiene porque el experimento final necesita E3."
        $summary | Export-Csv "logs\pipeline_summary.csv" -NoTypeInformation -Encoding UTF8
        exit $e3RetryExit
    }
}

# ============================================================
# 05. Compare and export
# ============================================================

$null = Run-Step `
    -Name "06 Compare experiments" `
    -Command "python scripts/06_compare_experiments.py --runs runs/" `
    -LogFile "logs\06_compare_experiments.log" `
    -Critical $true

$null = Run-Step `
    -Name "07 Export paper assets" `
    -Command "python scripts/07_export_paper_assets.py" `
    -LogFile "logs\07_export_paper_assets.log" `
    -Critical $true

# ============================================================
# 06. Final validation
# ============================================================

$null = Run-Step `
    -Name "08 Print final results" `
    -Command "python -c ""import pandas as pd; p='outputs/tables/final_results.csv'; df=pd.read_csv(p); print(df.to_string())""" `
    -LogFile "logs\08_final_results.log" `
    -Critical $true

$null = Run-Step `
    -Name "09 Validate four experiments" `
    -Command "python -c ""import pandas as pd; df=pd.read_csv('outputs/tables/final_results.csv'); ids=set(df['experiment_id']); print('Experiments:', ids); assert {'E0','E1','E2','E3'}.issubset(ids), f'Faltan experimentos: {set(['E0','E1','E2','E3'])-ids}'; print('OK: están E0, E1, E2 y E3')""" `
    -LogFile "logs\09_validate_four_experiments.log" `
    -Critical $true

$null = Run-Step `
    -Name "10 Count metrics files" `
    -Command "powershell -Command ""Get-ChildItem runs -Recurse -Filter metrics.json | Select-Object FullName""" `
    -LogFile "logs\10_metrics_files.log" `
    -Critical $false

$endTime = Get-Date
$totalDuration = New-TimeSpan -Start $startTime -End $endTime

$summary | Export-Csv "logs\pipeline_summary.csv" -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "========================================="
Write-Host "FIN PIPELINE FULL"
Write-Host "Inicio: $startTime"
Write-Host "Fin: $endTime"
Write-Host "Duración total: $($totalDuration.ToString())"
Write-Host "Resumen guardado en logs\pipeline_summary.csv"
Write-Host "========================================="
