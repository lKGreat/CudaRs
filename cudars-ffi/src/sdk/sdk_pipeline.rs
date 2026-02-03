use cudars_core::SdkErr;

use super::sdk_error::{clear_last_error, set_last_error};
use super::sdk_handles::{SdkPipelineHandle, SDK_PIPELINES};
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;

#[no_mangle]
pub extern "C" fn sdk_yolo_pipeline_run_image(
    pipeline: SdkPipelineHandle,
    data: *const u8,
    len: usize,
    out_meta: *mut SdkYoloPreprocessMeta,
) -> SdkErr {
    let mut pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get_mut(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    if let Some(ref mut gpu) = instance.yolo_gpu {
        let result = gpu.run_image(data, len, out_meta);
        if result == SdkErr::Ok {
            clear_last_error();
        }
        return result;
    }

    if let Some(ref mut cpu) = instance.yolo_cpu {
        let result = cpu.run_image(data, len, out_meta);
        if result == SdkErr::Ok {
            clear_last_error();
        }
        return result;
    }

    set_last_error("pipeline does not support yolo run");
    SdkErr::Unsupported
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_count(pipeline: SdkPipelineHandle, out_count: *mut usize) -> SdkErr {
    if out_count.is_null() {
        set_last_error("out_count is null");
        return SdkErr::InvalidArg;
    }

    let pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    let count = if let Some(ref gpu) = instance.yolo_gpu {
        gpu.output_count()
    } else if let Some(ref cpu) = instance.yolo_cpu {
        cpu.output_count()
    } else {
        0
    };

    unsafe {
        *out_count = count;
    }
    clear_last_error();
    SdkErr::Ok
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_shape_len(
    pipeline: SdkPipelineHandle,
    index: usize,
    out_len: *mut usize,
) -> SdkErr {
    if out_len.is_null() {
        set_last_error("out_len is null");
        return SdkErr::InvalidArg;
    }

    let pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    let len = if let Some(ref gpu) = instance.yolo_gpu {
        gpu.output_shape(index).map(|s| s.len()).unwrap_or(0)
    } else if let Some(ref cpu) = instance.yolo_cpu {
        cpu.output_count();
        0
    } else {
        0
    };

    if len == 0 {
        set_last_error("output shape not available");
        return SdkErr::InvalidArg;
    }

    unsafe {
        *out_len = len;
    }
    clear_last_error();
    SdkErr::Ok
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_shape_write(
    pipeline: SdkPipelineHandle,
    index: usize,
    dst: *mut i64,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    if dst.is_null() || out_written.is_null() {
        set_last_error("dst or out_written is null");
        return SdkErr::InvalidArg;
    }

    let pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    let shape = if let Some(ref gpu) = instance.yolo_gpu {
        gpu.output_shape(index)
    } else {
        None
    };

    let shape = match shape {
        Some(s) => s,
        None => {
            set_last_error("output shape not available");
            return SdkErr::InvalidArg;
        }
    };

    let to_copy = shape.len().min(cap);
    unsafe {
        std::ptr::copy_nonoverlapping(shape.as_ptr(), dst, to_copy);
        *out_written = to_copy;
    }

    if to_copy < shape.len() {
        set_last_error("shape buffer too small");
        return SdkErr::InvalidArg;
    }

    clear_last_error();
    SdkErr::Ok
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_bytes(
    pipeline: SdkPipelineHandle,
    index: usize,
    out_bytes: *mut usize,
) -> SdkErr {
    if out_bytes.is_null() {
        set_last_error("out_bytes is null");
        return SdkErr::InvalidArg;
    }

    let pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    let bytes = if let Some(ref gpu) = instance.yolo_gpu {
        gpu.output_bytes(index).unwrap_or(0)
    } else {
        0
    };

    if bytes == 0 {
        set_last_error("output bytes not available");
        return SdkErr::InvalidArg;
    }

    unsafe {
        *out_bytes = bytes;
    }
    clear_last_error();
    SdkErr::Ok
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_read_output(
    pipeline: SdkPipelineHandle,
    index: usize,
    dst: *mut u8,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    let pipelines = SDK_PIPELINES.lock().unwrap();
    let instance = match pipelines.get(pipeline) {
        Some(p) => p,
        None => {
            set_last_error("invalid pipeline handle");
            return SdkErr::InvalidArg;
        }
    };

    if let Some(ref gpu) = instance.yolo_gpu {
        let result = gpu.read_output(index, dst, cap, out_written);
        if result == SdkErr::Ok {
            clear_last_error();
        }
        return result;
    }

    set_last_error("pipeline does not support output read");
    SdkErr::Unsupported
}
