# PaddlePaddle to ONNX Model Converter (PowerShell)
# Batch converts PaddlePaddle models to ONNX format for use with OpenVINO

param(
    [Parameter(Mandatory=$false)]
    [string]$InputDir,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir,
    
    [Parameter(Mandatory=$false)]
    [string]$ModelDir,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$ModelFilename = "inference.json",
    
    [Parameter(Mandatory=$false)]
    [string]$ParamsFilename = "inference.pdiparams",
    
    [Parameter(Mandatory=$false)]
    [int]$OpsetVersion = 11,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoValidation,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

function Show-Help {
    Write-Host @"
PaddlePaddle to ONNX Model Converter

USAGE:
    .\convert_paddle_models.ps1 [OPTIONS]

OPTIONS:
    -ModelDir <path>         Directory containing a single PaddlePaddle model
    -OutputPath <path>       Path to save the converted ONNX model
    -InputDir <path>         Root directory for batch conversion
    -OutputDir <path>        Output root directory for batch conversion
    -ModelFilename <name>    Model structure filename (default: inference.json)
    -ParamsFilename <name>   Parameters filename (default: inference.pdiparams)
    -OpsetVersion <version>  ONNX opset version (default: 11)
    -NoValidation           Skip ONNX model validation
    -Help                   Show this help message

EXAMPLES:
    # Convert a single model
    .\convert_paddle_models.ps1 -ModelDir "E:\models\PP-OCRv5_mobile_det_infer" -OutputPath "model.onnx"

    # Batch convert multiple models
    .\convert_paddle_models.ps1 -InputDir "E:\models\paddle" -OutputDir "E:\models\onnx"

    # Convert with custom opset version
    .\convert_paddle_models.ps1 -ModelDir ".\model" -OutputPath "model.onnx" -OpsetVersion 13

REQUIREMENTS:
    - Python 3.7 or higher
    - paddle2onnx (pip install paddle2onnx)
    - onnx (pip install onnx)
"@
}

function Test-PythonPackage {
    param([string]$PackageName)
    
    $result = python -c "import $PackageName" 2>&1
    return $LASTEXITCODE -eq 0
}

function Check-Dependencies {
    Write-Host "Checking dependencies..." -ForegroundColor Cyan
    
    # Check Python
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Python not found. Please install Python 3.7 or higher." -ForegroundColor Red
        return $false
    }
    Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    
    # Check paddle2onnx
    if (-not (Test-PythonPackage "paddle2onnx")) {
        Write-Host "✗ paddle2onnx not found. Please install: pip install paddle2onnx" -ForegroundColor Red
        return $false
    }
    Write-Host "✓ paddle2onnx installed" -ForegroundColor Green
    
    # Check onnx
    if (-not (Test-PythonPackage "onnx")) {
        Write-Host "✗ onnx not found. Please install: pip install onnx" -ForegroundColor Red
        return $false
    }
    Write-Host "✓ onnx installed" -ForegroundColor Green
    
    return $true
}

function Convert-SingleModel {
    param(
        [string]$ModelDir,
        [string]$OutputPath,
        [string]$ModelFilename,
        [string]$ParamsFilename,
        [int]$OpsetVersion,
        [bool]$EnableValidation
    )
    
    Write-Host "`nConverting PaddlePaddle model to ONNX..." -ForegroundColor Cyan
    Write-Host "  Model dir: $ModelDir"
    Write-Host "  Output: $OutputPath"
    Write-Host "  Opset version: $OpsetVersion"
    
    # Check if model files exist
    $modelPath = Join-Path $ModelDir $ModelFilename
    $paramsPath = Join-Path $ModelDir $ParamsFilename
    
    if (-not (Test-Path $modelPath)) {
        Write-Host "✗ Model file not found: $modelPath" -ForegroundColor Red
        return $false
    }
    
    if (-not (Test-Path $paramsPath)) {
        Write-Host "✗ Parameters file not found: $paramsPath" -ForegroundColor Red
        return $false
    }
    
    # Build conversion command
    $scriptPath = Join-Path $PSScriptRoot "paddle2onnx_converter.py"
    
    if (-not (Test-Path $scriptPath)) {
        Write-Host "✗ Converter script not found: $scriptPath" -ForegroundColor Red
        return $false
    }
    
    $args = @(
        $scriptPath,
        "--model_dir", $ModelDir,
        "--output", $OutputPath,
        "--model_filename", $ModelFilename,
        "--params_filename", $ParamsFilename,
        "--opset_version", $OpsetVersion
    )
    
    if (-not $EnableValidation) {
        $args += "--no-validation"
    }
    
    # Run conversion
    & python $args
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Conversion successful!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Conversion failed!" -ForegroundColor Red
        return $false
    }
}

function Convert-BatchModels {
    param(
        [string]$InputDir,
        [string]$OutputDir,
        [string]$ModelFilename,
        [string]$ParamsFilename,
        [int]$OpsetVersion
    )
    
    Write-Host "`nSearching for PaddlePaddle models..." -ForegroundColor Cyan
    
    # Find all model directories
    $modelDirs = Get-ChildItem -Path $InputDir -Recurse -Filter $ModelFilename | 
                 Select-Object -ExpandProperty DirectoryName | 
                 Get-Unique
    
    if ($modelDirs.Count -eq 0) {
        Write-Host "No PaddlePaddle models found in $InputDir" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Found $($modelDirs.Count) model(s) to convert:" -ForegroundColor Cyan
    foreach ($dir in $modelDirs) {
        $relativePath = $dir.Replace($InputDir, "").TrimStart("\")
        Write-Host "  - $relativePath"
    }
    
    Write-Host "`n$('='*60)"
    
    # Convert each model
    $successCount = 0
    $totalCount = $modelDirs.Count
    
    for ($i = 0; $i -lt $totalCount; $i++) {
        $modelDir = $modelDirs[$i]
        $dirName = Split-Path $modelDir -Leaf
        
        Write-Host "`n[$($i+1)/$totalCount] Converting: $dirName" -ForegroundColor Cyan
        Write-Host $('-'*60)
        
        try {
            $relativePath = $modelDir.Replace($InputDir, "").TrimStart("\")
            $outputModelDir = Join-Path $OutputDir $relativePath
            $outputModelPath = Join-Path $outputModelDir "model.onnx"
            
            # Create output directory
            if (-not (Test-Path $outputModelDir)) {
                New-Item -ItemType Directory -Path $outputModelDir -Force | Out-Null
            }
            
            $result = Convert-SingleModel `
                -ModelDir $modelDir `
                -OutputPath $outputModelPath `
                -ModelFilename $ModelFilename `
                -ParamsFilename $ParamsFilename `
                -OpsetVersion $OpsetVersion `
                -EnableValidation $true
            
            if ($result) {
                $successCount++
            }
        }
        catch {
            Write-Host "Failed to convert ${dirName}: $_" -ForegroundColor Red
            continue
        }
    }
    
    Write-Host "`n$('='*60)"
    Write-Host "`nConversion complete: $successCount/$totalCount successful" -ForegroundColor Cyan
    
    if ($successCount -eq $totalCount) {
        Write-Host "All models converted successfully!" -ForegroundColor Green
    } elseif ($successCount -gt 0) {
        Write-Host "Some models failed to convert. Please check the errors above." -ForegroundColor Yellow
    } else {
        Write-Host "All conversions failed." -ForegroundColor Red
    }
}

# Main script
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "PaddlePaddle to ONNX Model Converter" -ForegroundColor Cyan
Write-Host $('='*60)

# Check dependencies
if (-not (Check-Dependencies)) {
    Write-Host "`nPlease install required dependencies and try again." -ForegroundColor Red
    exit 1
}

# Determine mode (single or batch)
$isBatchMode = $InputDir -and $OutputDir
$isSingleMode = $ModelDir -and $OutputPath

if (-not $isBatchMode -and -not $isSingleMode) {
    Write-Host "`nError: Invalid arguments." -ForegroundColor Red
    Write-Host "For single conversion: use -ModelDir and -OutputPath"
    Write-Host "For batch conversion: use -InputDir and -OutputDir"
    Write-Host "`nUse -Help for more information."
    exit 1
}

try {
    if ($isBatchMode) {
        # Batch conversion
        if (-not (Test-Path $InputDir)) {
            Write-Host "`nError: Input directory not found: $InputDir" -ForegroundColor Red
            exit 1
        }
        
        Convert-BatchModels `
            -InputDir $InputDir `
            -OutputDir $OutputDir `
            -ModelFilename $ModelFilename `
            -ParamsFilename $ParamsFilename `
            -OpsetVersion $OpsetVersion
    }
    else {
        # Single model conversion
        if (-not (Test-Path $ModelDir)) {
            Write-Host "`nError: Model directory not found: $ModelDir" -ForegroundColor Red
            exit 1
        }
        
        $result = Convert-SingleModel `
            -ModelDir $ModelDir `
            -OutputPath $OutputPath `
            -ModelFilename $ModelFilename `
            -ParamsFilename $ParamsFilename `
            -OpsetVersion $OpsetVersion `
            -EnableValidation (-not $NoValidation)
        
        if (-not $result) {
            exit 1
        }
    }
}
catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nDone!" -ForegroundColor Green
