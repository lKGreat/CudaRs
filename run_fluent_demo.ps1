$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Paddle/OpenCV/OpenVINO locations (adjust if you moved them)
$env:PADDLE_INFERENCE_ROOT = "E:\codeding\AI\paddle_inference"
$env:PADDLE_LIB = "E:\codeding\AI\paddle_inference\paddle\lib"
$env:OPENCV_DIR = "E:\codeding\AI\opencv\build"
$env:PADDLE_OCR_CPP_DIR = "E:\codeding\AI\PaddleOCR-3.3.2\deploy\cpp_infer"
$env:ABSL_INCLUDE = "C:\vcpkg\installed\x64-windows\include"
$env:ABSL_LIB = "C:\vcpkg\installed\x64-windows\lib"

# CudaRS FFI resolution
$env:CUDARS_FFI_DIR = Join-Path $repoRoot "dotnet\CudaRS.Examples\bin\Release\net8.0"

# DLL search path (merge all runtime deps)
$paths = @(
    $env:CUDARS_FFI_DIR,
    "E:\codeding\AI\paddle_inference\paddle\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\mklml\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\onednn\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\protobuf\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\glog\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\gflags\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\xxhash\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\zlib\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\onnxruntime\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\paddle2onnx\lib",
    "E:\codeding\AI\paddle_inference\third_party\install\openvino\intel64",
    "E:\codeding\AI\paddle_inference\third_party\install\tbb\lib",
    "E:\codeding\AI\cudars\target\release\build\onnxruntime-sys-4bfdfe0dbc709186\out\onnxruntime\onnxruntime-win-x64-1.8.1\lib",
    "E:\codeding\AI\cudars\openvino_env\Lib\site-packages\openvino\libs",
    "E:\codeding\AI\opencv\build\x64\vc16\bin",
    "C:\vcpkg\installed\x64-windows\bin",
    "C:\vcpkg\buildtrees\abseil\x64-windows-rel\bin"
)

$env:PATH = ($paths -join ";") + ";" + $env:PATH

Write-Host "== Build cudars-ffi (paddleocr + openvino + jpeg) =="
cargo build --release -p cudars-ffi --features "paddleocr openvino jpeg"

Write-Host "== Build .NET solution =="
dotnet build dotnet\CudaRS.sln -c Release

Write-Host "== Run Fluent demo =="
dotnet run --project dotnet\CudaRS.Examples\CudaRS.Examples.csproj -c Release
