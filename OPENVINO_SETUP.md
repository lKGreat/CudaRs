# OpenVINO 设置指南

## 当前状态

✅ **已完成**：
- PaddlePaddle 到 ONNX 转换工具（Python & PowerShell）
- C# Paddle2OnnxConverter 类
- 预处理配置解析器
- 完整示例代码
- 详细文档
- **Rust 编译错误已全部修复**

❌ **待完成**：
- 安装 OpenVINO 库

## 为什么需要 OpenVINO？

OpenVINO 用于：
1. 加载和运行 ONNX 模型（包括从 PaddlePaddle 转换的模型）
2. 提供 Intel CPU/GPU 优化的推理引擎
3. 支持动态shape、批量推理等高级功能

## 安装 OpenVINO

### 方法 1：下载预编译版本（推荐）

1. **下载 OpenVINO**
   - 访问：https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
   - 选择：Windows 版本
   - 下载：2024.x 或更新版本

2. **安装**
   ```bash
   # 解压到目录，例如：
   C:\Program Files (x86)\Intel\openvino_2024
   ```

3. **配置环境变量**
   ```powershell
   # 设置 INTEL_OPENVINO_DIR
   $env:INTEL_OPENVINO_DIR = "C:\Program Files (x86)\Intel\openvino_2024"
   
   # 添加到 PATH
   $env:PATH += ";$env:INTEL_OPENVINO_DIR\runtime\bin\intel64\Release"
   
   # 永久设置（PowerShell 管理员模式）
   [System.Environment]::SetEnvironmentVariable('INTEL_OPENVINO_DIR', 'C:\Program Files (x86)\Intel\openvino_2024', 'Machine')
   ```

4. **验证安装**
   ```bash
   # 检查 openvino.dll 是否存在
   dir "$env:INTEL_OPENVINO_DIR\runtime\bin\intel64\Release\openvino.dll"
   ```

### 方法 2：使用 vcpkg（开发者推荐）

```bash
# 安装 vcpkg（如果还没有）
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# 安装 OpenVINO
.\vcpkg install openvino:x64-windows
```

## 配置 Rust 构建

### 方法 1：环境变量配置

```powershell
# 设置 OpenVINO 库路径
$env:OPENVINO_INSTALL_DIR = "C:\Program Files (x86)\Intel\openvino_2024"
$env:LIB += ";$env:OPENVINO_INSTALL_DIR\runtime\lib\intel64\Release"
```

### 方法 2：在 build.rs 中配置

在 `cudars-ffi/build.rs` 中添加：

```rust
#[cfg(feature = "openvino")]
fn setup_openvino() {
    if let Ok(openvino_dir) = std::env::var("INTEL_OPENVINO_DIR") {
        println!("cargo:rustc-link-search=native={}/runtime/lib/intel64/Release", openvino_dir);
        println!("cargo:rustc-link-lib=openvino");
        println!("cargo:rustc-link-lib=openvino_c");
    } else {
        println!("cargo:warning=INTEL_OPENVINO_DIR not set, OpenVINO may not link");
    }
}
```

## 重新编译

安装 OpenVINO 后：

```bash
# 清理之前的构建
cargo clean -p cudars-ffi

# 重新构建（启用 OpenVINO）
cargo build -p cudars-ffi --release --features openvino

# 复制 DLL 到 .NET 项目
copy target\release\cudars_ffi.dll dotnet\CudaRS.Examples\bin\x64\Release\net8.0\
```

## 启用 OpenVINO 测试

在 `dotnet/CudaRS.Examples/Config.cs` 中：

```csharp
public const bool RunOpenVinoTests = true;  // 改为 true
```

## 测试 PaddlePaddle 模型转换

1. **准备模型**
   ```bash
   # 下载 PP-OCRv5 模型
   # https://github.com/PaddlePaddle/PaddleOCR
   ```

2. **安装转换工具**
   ```bash
   pip install paddle2onnx onnx onnxruntime
   ```

3. **转换模型**
   ```bash
   python scripts/paddle2onnx_converter.py \
     --model_dir E:\models\PP-OCRv5_mobile_det_infer \
     --output model.onnx
   ```

4. **运行测试**
   ```bash
   cd dotnet/CudaRS.Examples
   dotnet run
   ```

## 当前可用功能

即使没有 OpenVINO，您仍然可以使用：

### 1. PaddlePaddle 模型转换（不需要 OpenVINO）

```bash
# 转换工具已经可用
python scripts/paddle2onnx_converter.py --help
.\scripts\convert_paddle_models.ps1 -Help
```

### 2. YOLO 模型测试（使用其他后端）

如果有 TensorRT 或其他后端，可以直接测试 YOLO 模型：

```csharp
// 不使用 OpenVINO，使用 TensorRT
var pipeline = CudaRsFluent.Create()
    .Pipeline()
    .ForYolo(@"E:\codeding\AI\onnx\best\best.onnx", cfg => { ... })
    .AsTensorRT()  // 或 .AsPaddle()
    .BuildYoloFluent();
```

## 故障排除

### 链接错误：找不到 openvino.lib

**原因**：OpenVINO 未安装或路径未配置

**解决**：
1. 检查 OpenVINO 是否已安装
2. 设置环境变量 `INTEL_OPENVINO_DIR`
3. 将库路径添加到 `LIB` 环境变量

### 运行时错误：找不到 openvino.dll

**原因**：DLL 不在 PATH 中

**解决**：
```powershell
$env:PATH += ";C:\Program Files (x86)\Intel\openvino_2024\runtime\bin\intel64\Release"
```

### OpenVINO 版本不兼容

**推荐版本**：2023.x 或 2024.x

**检查版本**：
```bash
# 查看 OpenVINO 安装目录
dir "$env:INTEL_OPENVINO_DIR"
```

## 下一步

1. **不需要 OpenVINO**：继续使用转换脚本和文档
2. **需要 OpenVINO**：按照上述步骤安装并配置
3. **测试其他功能**：使用 TensorRT 或 PaddleOCR 后端

---

**参考资源**：
- [OpenVINO 官方文档](https://docs.openvino.ai/)
- [PaddlePaddle 转换指南](docs/PADDLE_OPENVINO_GUIDE.md)
- [快速开始](docs/PADDLE_QUICKSTART.md)
