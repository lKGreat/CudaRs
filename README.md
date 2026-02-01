# CudaRS

> Rust bindings for CUDA with C#/.NET interoperability

CudaRS 提供完整的 CUDA 生态系统绑定，从底层 FFI 到高级 .NET API。

## 项目结构

```
cudars/
├── build-support/cuda-build/    # 共享构建工具
├── cuda-sys/                    # FFI 绑定 (-sys crates)
│   ├── cuda-runtime-sys/        # CUDA Runtime API
│   ├── cuda-driver-sys/         # CUDA Driver API
│   ├── cublas-sys/              # cuBLAS
│   ├── cufft-sys/               # cuFFT
│   ├── curand-sys/              # cuRAND
│   ├── cusparse-sys/            # cuSPARSE
│   ├── cusolver-sys/            # cuSOLVER
│   ├── cudnn-sys/               # cuDNN
│   ├── nvrtc-sys/               # NVRTC
│   ├── nvjpeg-sys/              # nvJPEG
│   ├── npp-sys/                 # NPP
│   ├── cupti-sys/               # CUPTI
│   └── nvml-sys/                # NVML
├── cuda-rs/                     # 安全封装
│   ├── cuda-runtime/
│   ├── cuda-driver/
│   ├── cublas/
│   ├── cufft/
│   ├── curand/
│   ├── cusparse/
│   ├── cusolver/
│   ├── cudnn/
│   ├── nvrtc/
│   ├── nvjpeg/
│   ├── npp/
│   ├── cupti/
│   └── nvml/
├── cudars-ffi/                  # C 导出层 (cdylib)
└── dotnet/                      # C# 项目
    ├── CudaRS.Native/           # P/Invoke 绑定
    ├── CudaRS.Core/             # SafeHandle 封装
    ├── CudaRS/                  # 高级 API
    └── CudaRS.Examples/         # 示例程序
```

## 快速开始

### 前置条件

- CUDA Toolkit 11.x 或 12.x
- Rust 1.70+
- .NET 8.0+

### 构建 Rust 库

```bash
# 默认 CUDA 12
cargo build --release

# CUDA 11
cargo build --release --features cuda-11

# CUDA 12.3 特定版本
cargo build --release --features cuda-12-3
```

### 构建 .NET 项目

```bash
cd dotnet
dotnet build -c Release
```

### 运行示例

```bash
cd dotnet/CudaRS.Examples
dotnet run
```

## 使用方式

### C# 高级 API

```csharp
using CudaRS;

// 检查设备
Console.WriteLine($"CUDA Devices: {Cuda.DeviceCount}");

// 内存操作
float[] hostData = new float[] { 1, 2, 3, 4 };
using var deviceBuffer = hostData.ToDevice();

// 复制回主机
var result = deviceBuffer.ToArray();

// GPU 管理
var memInfo = GpuManagement.GetMemoryInfo(0);
Console.WriteLine($"GPU Memory: {memInfo.Used / 1024 / 1024} MB used");
```

### C# SafeHandle API

```csharp
using CudaRS.Core;

using var stream = new CudaStream();
using var cublasHandle = new CublasHandle();
cublasHandle.SetStream(stream);
```

### Rust 安全 API

```rust
use cuda_runtime::{Stream, DeviceBuffer};

let stream = Stream::new()?;
let mut buffer = DeviceBuffer::<f32>::new(1024)?;
buffer.memset(0)?;
stream.synchronize()?;
```

## 功能特性

| Feature Flag | 描述 |
|--------------|------|
| `cuda-11` | CUDA 11.x 支持 |
| `cuda-12` | CUDA 12.x 支持 (默认) |
| `cuda-12-3` | CUDA 12.3 特定 API |
| `runtime-linking` | 运行时动态链接 |

## 支持的库

- **CUDA Runtime** - 设备管理、内存、流、事件
- **CUDA Driver** - 底层驱动 API
- **cuBLAS** - 线性代数 (GEMM, AXPY 等)
- **cuFFT** - 快速傅里叶变换
- **cuRAND** - 随机数生成
- **cuSPARSE** - 稀疏矩阵运算
- **cuSOLVER** - 线性求解器
- **cuDNN** - 深度神经网络
- **NVRTC** - 运行时编译
- **nvJPEG** - JPEG 编解码
- **NPP** - 图像处理
- **CUPTI** - 性能分析
- **NVML** - GPU 管理

## 许可证

MIT OR Apache-2.0
