use cudars_core::SdkErr;

use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err};
use super::sdk_handles::{SdkPipelineHandle, SDK_PIPELINES};
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;
use super::paddleocr_output::SdkOcrLine;

#[no_mangle]
pub extern "C" fn sdk_yolo_pipeline_run_image(
    pipeline: SdkPipelineHandle,
    data: *const u8,
    len: usize,
    out_meta: *mut SdkYoloPreprocessMeta,
) -> SdkErr {
    with_panic_boundary_err("sdk_yolo_pipeline_run_image", || {
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
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

        #[cfg(feature = "openvino")]
        {
            if let Some(ref mut ov) = instance.yolo_openvino {
                let result = ov.run_image(data, len, out_meta);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }
        }

        set_last_error("pipeline does not support yolo run");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_tensor_pipeline_run(
    pipeline: SdkPipelineHandle,
    input: *const f32,
    input_len: usize,
    shape: *const i64,
    shape_len: usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_tensor_pipeline_run", || {
        if input.is_null() || shape.is_null() {
            set_last_error("input or shape is null");
            return SdkErr::InvalidArg;
        }

        #[cfg(not(feature = "openvino"))]
        {
            let _ = (pipeline, input, input_len, shape, shape_len);
        }

        #[cfg(feature = "openvino")]
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        #[cfg(feature = "openvino")]
        let instance = match pipelines.get_mut(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        #[cfg(feature = "openvino")]
        {
            if let Some(ref mut ov) = instance.openvino_tensor {
                let result = ov.run_tensor(input, input_len, shape, shape_len);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }
        }

        set_last_error("pipeline does not support tensor run");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_openvino_async_submit(
    pipeline: SdkPipelineHandle,
    input: *const f32,
    input_len: usize,
    shape: *const i64,
    shape_len: usize,
    out_request_id: *mut i32,
) -> SdkErr {
    with_panic_boundary_err("sdk_openvino_async_submit", || {
        if input.is_null() || shape.is_null() || out_request_id.is_null() {
            set_last_error("input, shape, or out_request_id is null");
            return SdkErr::InvalidArg;
        }

        #[cfg(not(feature = "openvino"))]
        {
            let _ = (pipeline, input, input_len, shape, shape_len, out_request_id);
        }

        #[cfg(feature = "openvino")]
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        #[cfg(feature = "openvino")]
        let instance = match pipelines.get_mut(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        #[cfg(feature = "openvino")]
        {
            if let Some(ref mut ov) = instance.openvino_tensor {
                let result = ov.submit_async(input, input_len, shape, shape_len, out_request_id);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }
        }

        set_last_error("pipeline does not support openvino async submit");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_openvino_async_wait(
    pipeline: SdkPipelineHandle,
    request_id: i32,
) -> SdkErr {
    with_panic_boundary_err("sdk_openvino_async_wait", || {
        #[cfg(not(feature = "openvino"))]
        {
            let _ = (pipeline, request_id);
        }

        #[cfg(feature = "openvino")]
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        #[cfg(feature = "openvino")]
        let instance = match pipelines.get_mut(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        #[cfg(feature = "openvino")]
        {
            if let Some(ref mut ov) = instance.openvino_tensor {
                let result = ov.wait_async(request_id);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }
        }

        set_last_error("pipeline does not support openvino async wait");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_count(pipeline: SdkPipelineHandle, out_count: *mut usize) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_get_output_count", || {
        if out_count.is_null() {
            set_last_error("out_count is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
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
            #[cfg(feature = "openvino")]
            {
                if let Some(ref ov) = instance.yolo_openvino {
                    ov.output_count()
                } else if let Some(ref tensor) = instance.openvino_tensor {
                    tensor.output_count()
                } else {
                    0
                }
            }
            #[cfg(not(feature = "openvino"))]
            {
                0
            }
        };

        unsafe {
            *out_count = count;
        }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_shape_len(
    pipeline: SdkPipelineHandle,
    index: usize,
    out_len: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_get_output_shape_len", || {
        if out_len.is_null() {
            set_last_error("out_len is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let len = if let Some(ref gpu) = instance.yolo_gpu {
            gpu.output_shape(index).map(|s: &[i64]| s.len()).unwrap_or(0)
        } else if let Some(ref cpu) = instance.yolo_cpu {
            cpu.output_shape(index).map(|s: &[i64]| s.len()).unwrap_or(0)
        } else {
            #[cfg(feature = "openvino")]
            {
                if let Some(ref ov) = instance.yolo_openvino {
                    ov.output_shape(index).map(|s: &[i64]| s.len()).unwrap_or(0)
                } else if let Some(ref tensor) = instance.openvino_tensor {
                    tensor.output_shape(index).map(|s: &[i64]| s.len()).unwrap_or(0)
                } else {
                    0
                }
            }
            #[cfg(not(feature = "openvino"))]
            {
                0
            }
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
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_shape_write(
    pipeline: SdkPipelineHandle,
    index: usize,
    dst: *mut i64,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_get_output_shape_write", || {
        if dst.is_null() || out_written.is_null() {
            set_last_error("dst or out_written is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let shape = if let Some(ref gpu) = instance.yolo_gpu {
            gpu.output_shape(index)
        } else if let Some(ref cpu) = instance.yolo_cpu {
            cpu.output_shape(index)
        } else {
            #[cfg(feature = "openvino")]
            {
                if let Some(ref ov) = instance.yolo_openvino {
                    ov.output_shape(index)
                } else if let Some(ref tensor) = instance.openvino_tensor {
                    tensor.output_shape(index)
                } else {
                    None
                }
            }
            #[cfg(not(feature = "openvino"))]
            {
                None
            }
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
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_get_output_bytes(
    pipeline: SdkPipelineHandle,
    index: usize,
    out_bytes: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_get_output_bytes", || {
        if out_bytes.is_null() {
            set_last_error("out_bytes is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let bytes = if let Some(ref gpu) = instance.yolo_gpu {
            gpu.output_bytes(index).unwrap_or(0)
        } else if let Some(ref cpu) = instance.yolo_cpu {
            cpu.output_bytes(index).unwrap_or(0)
        } else {
            #[cfg(feature = "openvino")]
            {
                if let Some(ref ov) = instance.yolo_openvino {
                    ov.output_bytes(index).unwrap_or(0)
                } else if let Some(ref tensor) = instance.openvino_tensor {
                    tensor.output_bytes(index).unwrap_or(0)
                } else {
                    0
                }
            }
            #[cfg(not(feature = "openvino"))]
            {
                0
            }
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
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_read_output(
    pipeline: SdkPipelineHandle,
    index: usize,
    dst: *mut u8,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_read_output", || {
        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
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

        if let Some(ref cpu) = instance.yolo_cpu {
            let result = cpu.read_output(index, dst, cap, out_written);
            if result == SdkErr::Ok {
                clear_last_error();
            }
            return result;
        }

        #[cfg(feature = "openvino")]
        {
            if let Some(ref ov) = instance.yolo_openvino {
                let result = ov.read_output(index, dst, cap, out_written);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }

            if let Some(ref tensor) = instance.openvino_tensor {
                let result = tensor.read_output(index, dst, cap, out_written);
                if result == SdkErr::Ok {
                    clear_last_error();
                }
                return result;
            }
        }

        set_last_error("pipeline does not support output read");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_run_image(
    pipeline: SdkPipelineHandle,
    data: *const u8,
    len: usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_run_image", || {
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get_mut(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        if let Some(ref mut ocr) = instance.paddleocr {
            let result = ocr.run_image(data, len);
            if result == SdkErr::Ok {
                clear_last_error();
            }
            return result;
        }

        set_last_error("pipeline does not support ocr run");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_get_line_count(
    pipeline: SdkPipelineHandle,
    out_count: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_get_line_count", || {
        if out_count.is_null() {
            set_last_error("out_count is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let count = if let Some(ref ocr) = instance.paddleocr {
            match ocr.line_count() {
                Ok(value) => value,
                Err(err) => return err,
            }
        } else {
            set_last_error("pipeline does not support ocr output");
            return SdkErr::Unsupported;
        };

        unsafe { *out_count = count; }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_write_lines(
    pipeline: SdkPipelineHandle,
    dst: *mut SdkOcrLine,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_write_lines", || {
        if dst.is_null() || out_written.is_null() {
            set_last_error("dst or out_written is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        if let Some(ref ocr) = instance.paddleocr {
            let result = ocr.write_lines(dst, cap, out_written);
            if result == SdkErr::Ok {
                clear_last_error();
            }
            return result;
        }

        set_last_error("pipeline does not support ocr output");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_get_text_bytes(
    pipeline: SdkPipelineHandle,
    out_bytes: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_get_text_bytes", || {
        if out_bytes.is_null() {
            set_last_error("out_bytes is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let bytes = if let Some(ref ocr) = instance.paddleocr {
            match ocr.text_bytes() {
                Ok(value) => value,
                Err(err) => return err,
            }
        } else {
            set_last_error("pipeline does not support ocr output");
            return SdkErr::Unsupported;
        };

        unsafe { *out_bytes = bytes; }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_write_text(
    pipeline: SdkPipelineHandle,
    dst: *mut i8,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_write_text", || {
        if dst.is_null() || out_written.is_null() {
            set_last_error("dst or out_written is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        if let Some(ref ocr) = instance.paddleocr {
            let result = ocr.write_text(dst, cap, out_written);
            if result == SdkErr::Ok {
                clear_last_error();
            }
            return result;
        }

        set_last_error("pipeline does not support ocr output");
        SdkErr::Unsupported
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_get_struct_json_bytes(
    pipeline: SdkPipelineHandle,
    out_bytes: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_get_struct_json_bytes", || {
        if out_bytes.is_null() {
            set_last_error("out_bytes is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        let bytes = if let Some(ref ocr) = instance.paddleocr {
            match ocr.struct_json_bytes() {
                Ok(value) => value,
                Err(err) => return err,
            }
        } else {
            set_last_error("pipeline does not support ocr output");
            return SdkErr::Unsupported;
        };

        unsafe { *out_bytes = bytes; }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_ocr_pipeline_write_struct_json(
    pipeline: SdkPipelineHandle,
    dst: *mut i8,
    cap: usize,
    out_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_ocr_pipeline_write_struct_json", || {
        if dst.is_null() || out_written.is_null() {
            set_last_error("dst or out_written is null");
            return SdkErr::InvalidArg;
        }

        let pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let instance = match pipelines.get(pipeline) {
            Some(p) => p,
            None => {
                set_last_error("invalid pipeline handle");
                return SdkErr::InvalidArg;
            }
        };

        if let Some(ref ocr) = instance.paddleocr {
            let result = ocr.write_struct_json(dst, cap, out_written);
            if result == SdkErr::Ok {
                clear_last_error();
            }
            return result;
        }

        set_last_error("pipeline does not support ocr output");
        SdkErr::Unsupported
    })
}

fn lock_pipelines(
) -> Result<std::sync::MutexGuard<'static, crate::runtime::HandleManager<super::pipeline_instance::PipelineInstance>>, SdkErr> {
    SDK_PIPELINES.lock().map_err(|_| {
        set_last_error("pipeline lock poisoned");
        SdkErr::Runtime
    })
}
