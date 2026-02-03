//! GPU-only image preprocessing for YOLO inference (fast path).
//!
//! This module implements a fused CUDA kernel that performs:
//! - Resize (bilinear)
//! - Letterbox padding (114)
//! - HWC(u8) -> CHW(f32)
//! - Normalize to [0,1]
//!
//! The kernel is JIT-compiled via NVRTC when the `rtc` feature is enabled.
//! When `rtc` is disabled, APIs return `CudaRsResult::ErrorNotSupported`.

use super::runtime::HandleManager;
use crate::runtime::{CudaRsEvent, CudaRsStream, EVENTS, STREAMS};
use super::CudaRsResult;
use cuda_runtime_sys::{
    cudaMemcpyAsync, cudaStreamCreate, cudaStreamDestroy, cudaStream_t,
};
use libc::{c_float, c_int, c_uchar, c_void, size_t};
#[cfg(feature = "rtc")]
use std::collections::HashMap;
use std::sync::{Arc, Mutex};


pub type CudaRsPreprocessHandle = u64;

lazy_static::lazy_static! {
    static ref PREPROCESS_CONTEXTS: Mutex<HandleManager<Arc<Mutex<PreprocessContext>>>> = Mutex::new(HandleManager::new());
}

#[cfg(feature = "rtc")]
lazy_static::lazy_static! {
    static ref KERNEL_CACHE: Mutex<HashMap<i32, Arc<Kernel>>> = Mutex::new(HashMap::new());
}

/// Preprocessing result with letterbox info.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaRsPreprocessResult {
    /// Pointer to output data on GPU (CHW, f32)
    pub output_ptr: *mut c_float,
    /// Output size in bytes
    pub output_size: size_t,
    /// Scale factor applied
    pub scale: c_float,
    /// X padding
    pub pad_x: c_int,
    /// Y padding
    pub pad_y: c_int,
    /// Original width
    pub original_width: c_int,
    /// Original height
    pub original_height: c_int,
}

pub struct PreprocessContext {
    #[allow(dead_code)]
    device_id: i32,

    /// Staging input buffer on device (HWC, u8)
    input_buffer: *mut c_uchar,
    input_capacity: usize,

    /// Output buffer on device (CHW, f32)
    output_buffer: *mut c_float,

    target_width: i32,
    target_height: i32,
    channels: i32,

    stream: cudaStream_t,
}

// Safety: This context owns CUDA resources. Access is synchronized behind a Mutex.
unsafe impl Send for PreprocessContext {}

impl Drop for PreprocessContext {
    fn drop(&mut self) {
        unsafe {
            if !self.input_buffer.is_null() {
                cuda_runtime_sys::cudaFree(self.input_buffer as *mut c_void);
            }
            if !self.output_buffer.is_null() {
                cuda_runtime_sys::cudaFree(self.output_buffer as *mut c_void);
            }
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

#[cfg(feature = "rtc")]
struct Kernel {
    _module: cuda_driver::Module,
    func: cuda_driver::Function,
}

#[cfg(feature = "rtc")]
fn get_or_create_kernel(device_id: i32) -> Result<Arc<Kernel>, CudaRsResult> {
    // Fast path: reuse.
    {
        let cache = KERNEL_CACHE.lock().unwrap();
        if let Some(k) = cache.get(&device_id) {
            return Ok(Arc::clone(k));
        }
    }

    // Slow path: compile.
    cuda_driver::init().map_err(|_| CudaRsResult::ErrorNotInitialized)?;

    let device = cuda_driver::Device::get(device_id).map_err(|_| CudaRsResult::ErrorInvalidValue)?;
    let (major, minor) = device.compute_capability().map_err(|_| CudaRsResult::ErrorUnknown)?;

    // NVRTC compile.
    let src = KERNEL_SRC;
    let prog = nvrtc::Program::new(src, "cudars_preprocess.cu").map_err(|_| CudaRsResult::ErrorUnknown)?;

    let arch = format!("--gpu-architecture=compute_{}{}", major, minor);
    let opts = [arch.as_str(), "--use_fast_math", "--std=c++14"];

    if let Err(_) = prog.compile(&opts) {
        // Helpful log in diag mode.
        if std::env::var("CUDARS_DIAG").as_deref() == Ok("1") {
            if let Ok(log) = prog.get_log() {
                eprintln!("[CudaRS] NVRTC compile log:\n{}", log);
            }
        }
        return Err(CudaRsResult::ErrorUnknown);
    }

    let ptx = prog.get_ptx().map_err(|_| CudaRsResult::ErrorUnknown)?;

    let module = cuda_driver::Module::load_data(&ptx).map_err(|_| CudaRsResult::ErrorUnknown)?;
    let func = module
        .get_function("cudars_letterbox_u8_to_chw_f32")
        .map_err(|_| CudaRsResult::ErrorUnknown)?;

    let kernel = Arc::new(Kernel { _module: module, func });

    let mut cache = KERNEL_CACHE.lock().unwrap();
    cache.insert(device_id, Arc::clone(&kernel));

    Ok(kernel)
}

#[cfg(feature = "rtc")]
const KERNEL_SRC: &str = r#"
extern \"C\" __global__ void cudars_letterbox_u8_to_chw_f32(
    const unsigned char* __restrict__ input,
    int in_w,
    int in_h,
    float* __restrict__ output,
    int out_w,
    int out_h,
    int channels,
    float scale,
    int pad_x,
    int pad_y,
    int new_w,
    int new_h
) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out_w || y >= out_h) return;

    // Default padding value for YOLO letterbox.
    const float pad_val = 114.0f / 255.0f;

    bool inside = (x >= pad_x) && (x < pad_x + new_w) && (y >= pad_y) && (y < pad_y + new_h);

    float r = pad_val, g = pad_val, b = pad_val;

    if (inside) {
        // Map output pixel -> input coordinate (bilinear).
        float src_x = ((float)(x - pad_x) + 0.5f) / scale - 0.5f;
        float src_y = ((float)(y - pad_y) + 0.5f) / scale - 0.5f;

        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = src_x - (float)x0;
        float fy = src_y - (float)y0;

        if (x0 < 0) { x0 = 0; fx = 0.0f; }
        if (y0 < 0) { y0 = 0; fy = 0.0f; }
        if (x1 >= in_w) { x1 = in_w - 1; }
        if (y1 >= in_h) { y1 = in_h - 1; }

        int idx00 = (y0 * in_w + x0) * channels;
        int idx01 = (y0 * in_w + x1) * channels;
        int idx10 = (y1 * in_w + x0) * channels;
        int idx11 = (y1 * in_w + x1) * channels;

        // Assume channels >= 3 (RGB/BGR). We keep order as-is.
        float c00_0 = (float)input[idx00 + 0];
        float c01_0 = (float)input[idx01 + 0];
        float c10_0 = (float)input[idx10 + 0];
        float c11_0 = (float)input[idx11 + 0];

        float c00_1 = (float)input[idx00 + 1];
        float c01_1 = (float)input[idx01 + 1];
        float c10_1 = (float)input[idx10 + 1];
        float c11_1 = (float)input[idx11 + 1];

        float c00_2 = (float)input[idx00 + 2];
        float c01_2 = (float)input[idx01 + 2];
        float c10_2 = (float)input[idx10 + 2];
        float c11_2 = (float)input[idx11 + 2];

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w01 = fx * (1.0f - fy);
        float w10 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        r = (c00_0 * w00 + c01_0 * w01 + c10_0 * w10 + c11_0 * w11) / 255.0f;
        g = (c00_1 * w00 + c01_1 * w01 + c10_1 * w10 + c11_1 * w11) / 255.0f;
        b = (c00_2 * w00 + c01_2 * w01 + c10_2 * w10 + c11_2 * w11) / 255.0f;
    }

    int hw = out_w * out_h;
    int out_idx = y * out_w + x;

    // CHW planar
    output[out_idx + 0 * hw] = r;
    if (channels > 1) output[out_idx + 1 * hw] = g;
    if (channels > 2) output[out_idx + 2 * hw] = b;
}
"#;

/// Create a preprocessing context with pre-allocated buffers.
#[no_mangle]
pub extern "C" fn cudars_preprocess_create(
    handle: *mut CudaRsPreprocessHandle,
    target_width: c_int,
    target_height: c_int,
    channels: c_int,
    max_input_width: c_int,
    max_input_height: c_int,
) -> CudaRsResult {
    if handle.is_null() || target_width <= 0 || target_height <= 0 || channels <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let mut dev: i32 = 0;
        if cuda_runtime_sys::cudaGetDevice(&mut dev) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        let mut stream: cudaStream_t = std::ptr::null_mut();
        if cudaStreamCreate(&mut stream) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        let input_capacity = (max_input_width * max_input_height * channels) as usize;
        let mut input_ptr: *mut c_void = std::ptr::null_mut();
        if cuda_runtime_sys::cudaMalloc(&mut input_ptr, input_capacity) != cuda_runtime_sys::cudaSuccess {
            cudaStreamDestroy(stream);
            return CudaRsResult::ErrorOutOfMemory;
        }

        let out_elems = (target_width * target_height * channels) as usize;
        let mut out_ptr: *mut c_void = std::ptr::null_mut();
        if cuda_runtime_sys::cudaMalloc(&mut out_ptr, out_elems * std::mem::size_of::<f32>()) != cuda_runtime_sys::cudaSuccess {
            cuda_runtime_sys::cudaFree(input_ptr);
            cudaStreamDestroy(stream);
            return CudaRsResult::ErrorOutOfMemory;
        }

        let ctx = PreprocessContext {
            device_id: dev,
            input_buffer: input_ptr as *mut c_uchar,
            input_capacity,
            output_buffer: out_ptr as *mut c_float,
            target_width,
            target_height,
            channels,
            stream,
        };

        let mut contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        let id = contexts.insert(Arc::new(Mutex::new(ctx)));
        *handle = id;

        CudaRsResult::Success
    }
}

#[no_mangle]
pub extern "C" fn cudars_preprocess_destroy(handle: CudaRsPreprocessHandle) -> CudaRsResult {
    let mut contexts = PREPROCESS_CONTEXTS.lock().unwrap();
    match contexts.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

#[no_mangle]
pub extern "C" fn cudars_preprocess_run(
    handle: CudaRsPreprocessHandle,
    input: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    if input.is_null() || result.is_null() || input_width <= 0 || input_height <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let ctx = ctx.lock().unwrap();

    let channels = ctx.channels;
    let in_bytes = (input_width * input_height * channels) as usize;
    if in_bytes > ctx.input_capacity {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        if cudaMemcpyAsync(
            ctx.input_buffer as *mut c_void,
            input as *const c_void,
            in_bytes,
            cuda_runtime_sys::cudaMemcpyHostToDevice,
            ctx.stream,
        ) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }
    }

    cudars_preprocess_run_device(handle, ctx.input_buffer, input_width, input_height, result)
}

/// Run preprocessing on a caller-provided stream (async).
///
/// If `done_event` is non-zero, the event will be recorded on the stream after the kernel.
#[no_mangle]
pub extern "C" fn cudars_preprocess_run_on_stream(
    handle: CudaRsPreprocessHandle,
    input: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    stream: CudaRsStream,
    done_event: CudaRsEvent,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    if input.is_null() || result.is_null() || input_width <= 0 || input_height <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let ctx = ctx.lock().unwrap();

    let channels = ctx.channels;
    let in_bytes = (input_width * input_height * channels) as usize;
    if in_bytes > ctx.input_capacity {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        if cudaMemcpyAsync(
            ctx.input_buffer as *mut c_void,
            input as *const c_void,
            in_bytes,
            cuda_runtime_sys::cudaMemcpyHostToDevice,
            stream_raw,
        ) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }
    }

    // Avoid calling into the device variant while holding locks.
    let staging_ptr = ctx.input_buffer;
    drop(ctx);

    cudars_preprocess_run_device_on_stream(handle, staging_ptr, input_width, input_height, stream, done_event, result)
}

fn preprocess_run_device_on_stream_impl(
    ctx: &mut PreprocessContext,
    input_device: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    stream_raw: cuda_runtime_sys::cudaStream_t,
    done_event: CudaRsEvent,
    output_device: *mut c_float,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    let target_w = ctx.target_width;
    let target_h = ctx.target_height;
    let channels = ctx.channels;

    // Letterbox parameters
    let scale = f32::min(target_w as f32 / input_width as f32, target_h as f32 / input_height as f32);
    let new_w = (input_width as f32 * scale).round() as i32;
    let new_h = (input_height as f32 * scale).round() as i32;
    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    let event_raw = if done_event != 0 {
        let events = EVENTS.lock().unwrap();
        let e = match events.get(done_event) {
            Some(e) => e,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        Some(e.as_raw())
    } else {
        None
    };

    #[cfg(not(feature = "rtc"))]
    {
        let _ = (
            scale,
            new_w,
            new_h,
            pad_x,
            pad_y,
            channels,
            stream_raw,
            output_device,
            event_raw,
            input_device,
            result,
        );
        return CudaRsResult::ErrorNotSupported;
    }

    #[cfg(feature = "rtc")]
    {
        let kernel = match get_or_create_kernel(ctx.device_id) {
            Ok(k) => k,
            Err(e) => return e,
        };

        // Launch on the provided stream (runtime stream is compatible with CUstream).
        let cu_stream = stream_raw as cuda_driver_sys::CUstream;

        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((target_w as u32) + block.0 - 1) / block.0,
            ((target_h as u32) + block.1 - 1) / block.1,
            1u32,
        );

        unsafe {
            let mut p_input = input_device as *const u8;
            let mut p_in_w = input_width;
            let mut p_in_h = input_height;
            let mut p_output = output_device;
            let mut p_out_w = target_w;
            let mut p_out_h = target_h;
            let mut p_channels = channels;
            let mut p_scale = scale;
            let mut p_pad_x = pad_x;
            let mut p_pad_y = pad_y;
            let mut p_new_w = new_w;
            let mut p_new_h = new_h;

            let mut params: [*mut c_void; 12] = [
                (&mut p_input as *mut _ as *mut c_void),
                (&mut p_in_w as *mut _ as *mut c_void),
                (&mut p_in_h as *mut _ as *mut c_void),
                (&mut p_output as *mut _ as *mut c_void),
                (&mut p_out_w as *mut _ as *mut c_void),
                (&mut p_out_h as *mut _ as *mut c_void),
                (&mut p_channels as *mut _ as *mut c_void),
                (&mut p_scale as *mut _ as *mut c_void),
                (&mut p_pad_x as *mut _ as *mut c_void),
                (&mut p_pad_y as *mut _ as *mut c_void),
                (&mut p_new_w as *mut _ as *mut c_void),
                (&mut p_new_h as *mut _ as *mut c_void),
            ];

            if let Err(_) = kernel.func.launch(grid, block, 0, cu_stream, &mut params) {
                return CudaRsResult::ErrorUnknown;
            }

            if let Some(ev) = event_raw {
                let code = cuda_runtime_sys::cudaEventRecord(ev, stream_raw);
                if code != cuda_runtime_sys::cudaSuccess {
                    return CudaRsResult::ErrorUnknown;
                }
            }

            (*result).output_ptr = output_device;
            (*result).output_size =
                (target_w * target_h * channels) as usize * std::mem::size_of::<f32>();
            (*result).scale = scale;
            (*result).pad_x = pad_x;
            (*result).pad_y = pad_y;
            (*result).original_width = input_width;
            (*result).original_height = input_height;
        }

        CudaRsResult::Success
    }
}

/// Run preprocessing (device input) on a caller-provided stream (async).
///
/// If `done_event` is non-zero, the event will be recorded on the stream after the kernel.
#[no_mangle]
pub extern "C" fn cudars_preprocess_run_device_on_stream(
    handle: CudaRsPreprocessHandle,
    input_device: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    stream: CudaRsStream,
    done_event: CudaRsEvent,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    if input_device.is_null() || result.is_null() || input_width <= 0 || input_height <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let mut ctx = ctx.lock().unwrap();

    let output_buffer = ctx.output_buffer;

    preprocess_run_device_on_stream_impl(
        &mut ctx,
        input_device,
        input_width,
        input_height,
        stream_raw,
        done_event,
        output_buffer,
        result,
    )
}

/// Run preprocessing (device input) on a caller-provided stream into a caller-provided output buffer (async).
///
/// If `done_event` is non-zero, the event will be recorded on the stream after the kernel.
#[no_mangle]
pub extern "C" fn cudars_preprocess_run_device_on_stream_into(
    handle: CudaRsPreprocessHandle,
    input_device: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    stream: CudaRsStream,
    done_event: CudaRsEvent,
    output_device: *mut c_float,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    if input_device.is_null() || output_device.is_null() || result.is_null() || input_width <= 0 || input_height <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let mut ctx = ctx.lock().unwrap();

    preprocess_run_device_on_stream_impl(
        &mut ctx,
        input_device,
        input_width,
        input_height,
        stream_raw,
        done_event,
        output_device,
        result,
    )
}

#[no_mangle]
pub extern "C" fn cudars_preprocess_run_device(
    handle: CudaRsPreprocessHandle,
    input_device: *const c_uchar,
    input_width: c_int,
    input_height: c_int,
    result: *mut CudaRsPreprocessResult,
) -> CudaRsResult {
    if input_device.is_null() || result.is_null() || input_width <= 0 || input_height <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let ctx = ctx.lock().unwrap();

    let target_w = ctx.target_width;
    let target_h = ctx.target_height;
    let channels = ctx.channels;

    // Letterbox parameters
    let scale = f32::min(target_w as f32 / input_width as f32, target_h as f32 / input_height as f32);
    let new_w = (input_width as f32 * scale).round() as i32;
    let new_h = (input_height as f32 * scale).round() as i32;
    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    #[cfg(not(feature = "rtc"))]
    {
        let _ = (scale, new_w, new_h, pad_x, pad_y, channels);
        return CudaRsResult::ErrorNotSupported;
    }

    #[cfg(feature = "rtc")]
    {
        let kernel = match get_or_create_kernel(ctx.device_id) {
            Ok(k) => k,
            Err(e) => return e,
        };

        // Launch on the same stream (runtime stream is compatible with CUstream).
        let stream = ctx.stream as cuda_driver_sys::CUstream;

        let block = (16u32, 16u32, 1u32);
        let grid = (
            ((target_w as u32) + block.0 - 1) / block.0,
            ((target_h as u32) + block.1 - 1) / block.1,
            1u32,
        );

        unsafe {
            let mut p_input = input_device as *const u8;
            let mut p_in_w = input_width;
            let mut p_in_h = input_height;
            let mut p_output = ctx.output_buffer;
            let mut p_out_w = target_w;
            let mut p_out_h = target_h;
            let mut p_channels = channels;
            let mut p_scale = scale;
            let mut p_pad_x = pad_x;
            let mut p_pad_y = pad_y;
            let mut p_new_w = new_w;
            let mut p_new_h = new_h;

            let mut params: [*mut c_void; 12] = [
                (&mut p_input as *mut _ as *mut c_void),
                (&mut p_in_w as *mut _ as *mut c_void),
                (&mut p_in_h as *mut _ as *mut c_void),
                (&mut p_output as *mut _ as *mut c_void),
                (&mut p_out_w as *mut _ as *mut c_void),
                (&mut p_out_h as *mut _ as *mut c_void),
                (&mut p_channels as *mut _ as *mut c_void),
                (&mut p_scale as *mut _ as *mut c_void),
                (&mut p_pad_x as *mut _ as *mut c_void),
                (&mut p_pad_y as *mut _ as *mut c_void),
                (&mut p_new_w as *mut _ as *mut c_void),
                (&mut p_new_h as *mut _ as *mut c_void),
            ];

            if let Err(_) = kernel.func.launch(grid, block, 0, stream, &mut params) {
                return CudaRsResult::ErrorUnknown;
            }

            if cuda_runtime_sys::cudaStreamSynchronize(ctx.stream) != cuda_runtime_sys::cudaSuccess {
                return CudaRsResult::ErrorUnknown;
            }

            (*result).output_ptr = ctx.output_buffer;
            (*result).output_size = (target_w * target_h * channels) as usize * std::mem::size_of::<f32>();
            (*result).scale = scale;
            (*result).pad_x = pad_x;
            (*result).pad_y = pad_y;
            (*result).original_width = input_width;
            (*result).original_height = input_height;
        }

        CudaRsResult::Success
    }
}

#[no_mangle]
pub extern "C" fn cudars_preprocess_get_output_ptr(
    handle: CudaRsPreprocessHandle,
    output: *mut *mut c_float,
) -> CudaRsResult {
    if output.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let ctx = {
        let contexts = PREPROCESS_CONTEXTS.lock().unwrap();
        match contexts.get(handle) {
            Some(c) => Arc::clone(c),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let ctx = ctx.lock().unwrap();

    unsafe { *output = ctx.output_buffer };
    CudaRsResult::Success
}
