# FFI Contract (C ABI)

This document defines the stable C ABI for CudaRS. All non-Rust languages must use the `sdk_*` C ABI and must not rely on Rust ABI.

## Naming and ABI Rules

- All exported symbols use `extern "C"` and stable names (no mangling).
- Public structs/enums are `#[repr(C)]`.
- No Rust generics, `Vec`, `String`, `Result`, or trait objects across the ABI.
- UTF-8 strings are passed as `(ptr, len)` without a trailing NUL.

## Error Model

All functions return `SdkErr` and never unwind across the boundary. Use `sdk_last_error_message_utf8` for the last error message (thread-local).

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

## Versioning

```c
uint32_t sdk_abi_version(void);
const char* sdk_version_string(void);
SdkErr sdk_version_string_len(size_t* out_len);
SdkErr sdk_version_string_write(char* buf, size_t cap, size_t* out_written);
```

## Handles

Handles are opaque `uint64_t` values returned by creation APIs and released via the corresponding destroy function. They must not be freed across language boundaries.

## Model and Pipeline Specs

```c
typedef enum { SDK_MODEL_UNKNOWN = 0, SDK_MODEL_YOLO = 1, SDK_MODEL_PADDLE_OCR = 2, SDK_MODEL_OPENVINO = 3 } SdkModelKind;
typedef enum {
  SDK_PIPELINE_UNKNOWN = 0,
  SDK_PIPELINE_YOLO_CPU = 1,
  SDK_PIPELINE_YOLO_GPU_THROUGHPUT = 2,
  SDK_PIPELINE_PADDLE_OCR = 3,
  SDK_PIPELINE_YOLO_OPENVINO = 4,
  SDK_PIPELINE_OPENVINO_TENSOR = 5
} SdkPipelineKind;

typedef struct {
  const char* id_ptr;
  size_t id_len;
  SdkModelKind kind;
  const char* config_json_ptr;
  size_t config_json_len;
} SdkModelSpec;

typedef struct {
  const char* id_ptr;
  size_t id_len;
  SdkPipelineKind kind;
  const char* config_json_ptr;
  size_t config_json_len;
} SdkPipelineSpec;
```

## Model Manager and Pipelines

```c
SdkErr sdk_model_manager_create(uint64_t* out_handle);
SdkErr sdk_model_manager_destroy(uint64_t handle);
SdkErr sdk_model_manager_load_model(uint64_t manager, const SdkModelSpec* spec, uint64_t* out_model);
SdkErr sdk_model_create_pipeline(uint64_t model, const SdkPipelineSpec* spec, uint64_t* out_pipeline);
SdkErr sdk_pipeline_destroy(uint64_t pipeline);
```

## YOLO Pipeline Execution

```c
typedef struct {
  float scale;
  int32_t pad_x;
  int32_t pad_y;
  int32_t original_width;
  int32_t original_height;
} SdkYoloPreprocessMeta;

SdkErr sdk_yolo_pipeline_run_image(
  uint64_t pipeline,
  const uint8_t* data,
  size_t len,
  SdkYoloPreprocessMeta* out_meta);

// Generic tensor pipeline execution (OpenVINO).
SdkErr sdk_tensor_pipeline_run(
  uint64_t pipeline,
  const float* input,
  size_t input_len,
  const int64_t* shape,
  size_t shape_len);
```

## PaddleOCR Pipeline Execution

```c
typedef struct {
  float points[8];
  float score;
  int32_t cls_label;
  float cls_score;
  uint32_t text_offset;
  uint32_t text_len;
} SdkOcrLine;

SdkErr sdk_ocr_pipeline_run_image(uint64_t pipeline, const uint8_t* data, size_t len);
SdkErr sdk_ocr_pipeline_get_line_count(uint64_t pipeline, size_t* out_count);
SdkErr sdk_ocr_pipeline_write_lines(uint64_t pipeline, SdkOcrLine* dst, size_t cap, size_t* out_written);
SdkErr sdk_ocr_pipeline_get_text_bytes(uint64_t pipeline, size_t* out_bytes);
SdkErr sdk_ocr_pipeline_write_text(uint64_t pipeline, char* dst, size_t cap, size_t* out_written);
SdkErr sdk_ocr_pipeline_get_struct_json_bytes(uint64_t pipeline, size_t* out_bytes);
SdkErr sdk_ocr_pipeline_write_struct_json(uint64_t pipeline, char* dst, size_t cap, size_t* out_written);
```

## Output Retrieval (No SDK Allocations)

The caller owns output buffers and queries required sizes via the following helpers:

```c
SdkErr sdk_pipeline_get_output_count(uint64_t pipeline, size_t* out_count);
SdkErr sdk_pipeline_get_output_shape_len(uint64_t pipeline, size_t index, size_t* out_len);
SdkErr sdk_pipeline_get_output_shape_write(uint64_t pipeline, size_t index, int64_t* dst, size_t cap, size_t* out_written);
SdkErr sdk_pipeline_get_output_bytes(uint64_t pipeline, size_t index, size_t* out_bytes);
SdkErr sdk_pipeline_read_output(uint64_t pipeline, size_t index, uint8_t* dst, size_t cap, size_t* out_written);
```

## Tooling

- `sdk.h` is generated via `cbindgen` into `cudars-ffi/include/sdk.h`.
- C# P/Invoke is generated via `csbindgen` (or maintained manually for the curated surface).
