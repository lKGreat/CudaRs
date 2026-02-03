# FFI Contract (C ABI)

本文件定义本仓库对外唯一稳定契约：**C ABI**。C++/Java/JNI/C#/PInvoke/Python 仅做薄封装，禁止直接依赖 Rust ABI。

## 1. ABI 稳定性与符号
- 对外导出必须使用 `extern "C"` 且符号稳定（`#[no_mangle]` 或 `#[unsafe(no_mangle)]`，视 toolchain 规则）。
- 禁止在 ABI 层公开 Rust 泛型、trait object、`Result`、`String`、`Vec` 等非 FFI-safe 类型。
- 所有对外结构体/枚举必须 `#[repr(C)]` 或显式整数 repr。

## 2. 句柄与生命周期
- 对外仅暴露 **opaque handle**（如 `SdkHandle*`），由 `create/destroy` 成对管理。
- 句柄不可被外部语言直接解引用；禁止跨库 free。

**示例**
```c
typedef struct SdkHandle SdkHandle;

SdkErr sdk_create(SdkHandle** out);
SdkErr sdk_destroy(SdkHandle* handle);
```

## 3. 内存/字符串/缓冲区规范
统一两套模式，避免跨语言 free：

1) **调用方分配**
```c
SdkErr sdk_get_name_len(SdkHandle* h, size_t* out_len);
SdkErr sdk_get_name_write(SdkHandle* h, char* buf, size_t cap, size_t* out_written);
```

2) **SDK 分配 + SDK 释放**
```c
SdkErr sdk_get_blob(SdkHandle* h, uint8_t** out_ptr, size_t* out_len);
SdkErr sdk_free(void* ptr, size_t len);
```

约定：字符串均为 UTF-8，长度不含 `\0`；如需 NUL 结尾在文档中明确。

## 4. 错误模型
- 所有 FFI 函数返回 `SdkErr` 错误码，**不**跨边界抛异常/`panic`。
- 错误详情通过 `sdk_last_error_message_utf8()` 获取（建议线程本地存储）。

**示例**
```c
typedef enum {
  SDK_OK = 0,
  SDK_ERR_INVALID_ARG = 1,
  SDK_ERR_OUT_OF_MEMORY = 2,
  SDK_ERR_RUNTIME = 3,
  SDK_ERR_UNSUPPORTED = 4,
  SDK_ERR_NOT_FOUND = 5,
  SDK_ERR_TIMEOUT = 6,
  SDK_ERR_BUSY = 7,
  SDK_ERR_IO = 8,
  SDK_ERR_PERMISSION = 9,
  SDK_ERR_CANCELED = 10,
  SDK_ERR_BAD_STATE = 11,
  SDK_ERR_VERSION_MISMATCH = 12,
  SDK_ERR_BACKEND = 13
} SdkErr;

SdkErr sdk_last_error_message_utf8(const char** out_ptr, size_t* out_len);
```

**错误码表（规范化语义）**
- `SDK_OK`：成功。
- `SDK_ERR_INVALID_ARG`：参数为空、越界、或违反契约。
- `SDK_ERR_OUT_OF_MEMORY`：内存或显存不足。
- `SDK_ERR_RUNTIME`：未归类运行时错误（含 panic 捕获）。
- `SDK_ERR_UNSUPPORTED`：当前平台/设备/版本不支持该能力。
- `SDK_ERR_NOT_FOUND`：资源不存在（如模型/设备/键）。
- `SDK_ERR_TIMEOUT`：超时（可重试）。
- `SDK_ERR_BUSY`：资源忙（可重试）。
- `SDK_ERR_IO`：I/O 失败（文件/网络/驱动）。
- `SDK_ERR_PERMISSION`：权限不足或访问被拒。
- `SDK_ERR_CANCELED`：被取消（调用方或 SDK 触发）。
- `SDK_ERR_BAD_STATE`：状态不允许（需先调用某 API）。
- `SDK_ERR_VERSION_MISMATCH`：ABI 或数据版本不匹配。
- `SDK_ERR_BACKEND`：后端库/驱动返回的错误（CUDA/ONNX/TensorRT 等）。

**补充约定**
- 所有失败均应设置线程本地错误消息；成功调用会清空上一次错误消息。
- `sdk_last_error_message_utf8()` 返回的指针只在当前线程、下次 SDK 调用前有效。

## 5. Panic/异常边界
- Rust 入口统一 `catch_unwind`，将 panic 映射为 `SDK_ERR_RUNTIME`。
- C++ 异常不得穿越 FFI 边界；在 C++ wrapper 内部处理。

## 6. 版本与兼容性
- 提供 `sdk_abi_version()`（仅 ABI breaking 时递增）与 `sdk_version_string()`。
- 头文件必须说明线程安全、可重入性、回调线程模型。

## 7. 运行时模型（Runtime）
- `sdk-core` 保持 runtime-agnostic（Sans-IO / 依赖注入）。
- FFI 层自行选择/内置 runtime（feature 切换），对外 API 以 **阻塞式** 为默认，异步/回调为可选增强。

## 8. 线程与回调规则（必须在头文件注明）
- **线程安全级别**：每个 handle 标注 `thread-safe` 或 `thread-confined`。默认不跨线程共享，除非明确声明。
- **回调注册**：提供 `register/unregister`；回调携带 `user_data` 指针（由调用方管理生命周期）。
- **回调线程**：明确回调执行线程（SDK 内部线程池/调用线程/外部 runtime）。不保证在调用线程执行，除非文档声明。
- **回调可重入性**：回调不得调用会导致同一 handle 死锁的 API；如允许，需文档说明。
- **销毁语义**：`destroy` 之前必须先 `unregister` 或停止回调；`destroy` 后不得再触发回调。
- **阻塞/耗时**：回调内禁止长时间阻塞；必要时由调用方自行异步转发。
- **取消**：异步/长任务提供 `cancel` 句柄，返回 `SDK_ERR_CANCELED`。

## 8. 产物与自动生成
- `sdk.h`：由 `cbindgen` 从 Rust 导出自动生成。
- C ABI 库：使用 `cargo-c` 输出 `.dll/.so/.dylib` + `pkg-config`。
- C# P/Invoke：使用 `csbindgen` 生成 `[DllImport]` 声明。

## 9. 语言侧薄封装规则（摘要）
- **C++**：仅封 C API；RAII 封装 handle（`unique_ptr` + deleter），错误码转 `std::error_code` 或异常。
- **Java/JNI**：JNI glue 只做映射；`JNIEnv*` 不跨线程；必要时 `AttachCurrentThread/DetachCurrentThread`。
- **C#**：使用 `SafeHandle` 管理释放；调用方分配缓冲区优先。
- **Python**：优先 `ctypes`/`cffi` 封 C ABI；需要 Pythonic API 时再考虑 PyO3。

## 10. 命名约定
- 导出函数统一前缀 `sdk_`，类型使用 `Sdk*`。
- 新增 API 时，先写 C 头文件签名，再实现 Rust 导出。

## 11. C# 封装命名与内存规范（强制）
**命名边界**
- 公共 API 只使用通用语义名（`Client/Session/Request/Response/Options/Config/Handle/Buffer/Span`），不出现业务域名词。
- 业务域名词仅允许出现在内部命名空间 `YourSdk.Internal.<Biz>` 或独立扩展包 `YourSdk.<Biz>`。
- 指针/不安全能力必须显式标注并隔离在 `YourSdk.Unsafe`。

**命名空间固定分层**
- `YourSdk`：对外 OO API。
- `YourSdk.Interop`：`SafeHandle`、`NativeBuffer`、UTF-8 工具、错误翻译。
- `YourSdk.Native`：P/Invoke / `LibraryImport` 入口（internal）。
- `YourSdk.Unsafe`：指针型/Raw API（可 public，但必须显式标注危险性）。

**类型命名**
- 资源/句柄：`SdkClient` / `SdkSession`；对应 `SafeHandle` 为 `ClientHandle` / `SessionHandle`。
- 配置/构建：`SdkOptions` / `SdkConfig`；构建器后缀固定为 `Builder`（如 `SdkClientBuilder`）。
- 数据载体：`NativeSlice` / `ByteSlice`（view），`NativeBuffer`（拥有型、`IDisposable`）。
- UTF-8：`Utf8Z`（NUL 结尾）、`Utf8Span`（view）、`Utf8Buffer`（拥有型）。
- 错误：`SdkErrorCode`、`SdkException`、可选 `SdkStatus`。

**方法命名与缓冲区模板**
- OO 层动词 + 宾语；异步方法后缀 `Async`；`TryGetXxx` 返回 `bool`。
- 输出缓冲区统一模板：
  - `GetFoo(Span<byte> destination, out int bytesWritten)`
  - `GetFooRaw(byte* dst, nuint dstCapacity, out nuint bytesWritten)`（放在 `YourSdk.Unsafe`）
- 参数命名固定：`src/dst` 或 `input/output`；长度用 `Length`，容量用 `Capacity`；字节数量用 `bytesWritten/bytesRead`。

**P/Invoke 规则**
- C 导出名保持 `snake_case`，C# 入口用 `PascalCase` 并指定 `EntryPoint`。
- 优先 `LibraryImport` + 明确 UTF-8 marshalling；避免默认字符串编组。

**内存与性能约束**
- 对外优先 `Span<byte>`，避免暴露 `IntPtr`；仅在 `Unsafe` 提供指针重载。
- `SuppressGCTransition` 仅用于超短、无阻塞、无回调函数，且需要明确注释与基准验证。

## 12. 质量门禁（零告警）
- Rust 与 C# 必须 **0 warnings、0 errors**；禁止引入未使用变量或未使用的设计痕迹。
- CI 强制：Rust 使用 `-D warnings`；C# 使用 `TreatWarningsAsErrors=true`。
