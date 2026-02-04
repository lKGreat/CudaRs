use libc::c_void;

use cudars_core::SdkErr;

use crate::sdk::sdk_error::set_last_error;
use crate::sdk::paddleocr_model_config::PaddleOcrModelConfig;
use crate::sdk::paddleocr_pipeline_config::PaddleOcrPipelineConfig;
use crate::sdk::paddleocr_output::SdkOcrLine;

#[cfg(feature = "paddleocr")]
use crate::paddleocr::{
    paddleocr_create, paddleocr_destroy, paddleocr_get_line_count,
    paddleocr_get_struct_json_bytes, paddleocr_get_text_bytes, paddleocr_last_error,
    paddleocr_run_image, paddleocr_write_lines, paddleocr_write_struct_json,
    paddleocr_write_text, PaddleOcrInitOptions, PaddleOcrLine, PADDLEOCR_OPTION_UNSET_I32,
};

#[cfg(feature = "paddleocr")]
use std::ffi::CString;

pub struct PaddleOcrPipeline {
    #[cfg(feature = "paddleocr")]
    handle: *mut c_void,
    #[cfg(not(feature = "paddleocr"))]
    _stub: (),
    enable_struct_json: bool,
}

unsafe impl Send for PaddleOcrPipeline {}

impl PaddleOcrPipeline {
    pub fn new(model: &PaddleOcrModelConfig, pipeline: &PaddleOcrPipelineConfig) -> Result<Self, SdkErr> {
        if model.det_model_dir.is_empty() || model.rec_model_dir.is_empty() {
            set_last_error("det_model_dir and rec_model_dir are required");
            return Err(SdkErr::InvalidArg);
        }

        #[cfg(not(feature = "paddleocr"))]
        {
            let _ = model;
            let _ = pipeline;
            set_last_error("paddleocr feature not enabled");
            return Err(SdkErr::Unsupported);
        }

        #[cfg(feature = "paddleocr")]
        {
            let mut cstrings = Vec::new();
            fn push_opt(cstrings: &mut Vec<CString>, value: &Option<String>) -> *const i8 {
                match value {
                    Some(v) if !v.is_empty() => {
                        let c = CString::new(v.as_str()).ok();
                        if let Some(c) = c {
                            cstrings.push(c);
                            return cstrings.last().unwrap().as_ptr();
                        }
                        std::ptr::null()
                    }
                    _ => std::ptr::null(),
                }
            }
            fn push_str(cstrings: &mut Vec<CString>, value: &str) -> *const i8 {
                if value.is_empty() {
                    return std::ptr::null();
                }
                let c = CString::new(value).ok();
                if let Some(c) = c {
                    cstrings.push(c);
                    return cstrings.last().unwrap().as_ptr();
                }
                std::ptr::null()
            }

            let device = model.device.clone().unwrap_or_else(|| "cpu".to_string());
            let precision = model.precision.clone().unwrap_or_else(|| "fp32".to_string());

            let mut options = PaddleOcrInitOptions {
                doc_orientation_model_name: push_opt(&mut cstrings, &model.doc_orientation_model_name),
                doc_orientation_model_dir: push_opt(&mut cstrings, &model.doc_orientation_model_dir),
                doc_unwarping_model_name: push_opt(&mut cstrings, &model.doc_unwarping_model_name),
                doc_unwarping_model_dir: push_opt(&mut cstrings, &model.doc_unwarping_model_dir),
                text_detection_model_name: push_opt(&mut cstrings, &model.text_detection_model_name),
                text_detection_model_dir: push_str(&mut cstrings, &model.det_model_dir),
                textline_orientation_model_name: push_opt(&mut cstrings, &model.textline_orientation_model_name),
                textline_orientation_model_dir: push_opt(&mut cstrings, &model.textline_orientation_model_dir),
                text_recognition_model_name: push_opt(&mut cstrings, &model.text_recognition_model_name),
                text_recognition_model_dir: push_str(&mut cstrings, &model.rec_model_dir),
                lang: push_opt(&mut cstrings, &model.lang),
                ocr_version: push_opt(&mut cstrings, &model.ocr_version),
                vis_font_dir: push_opt(&mut cstrings, &model.vis_font_dir),
                device: push_str(&mut cstrings, &device),
                precision: push_str(&mut cstrings, &precision),
                text_det_limit_type: push_opt(&mut cstrings, &model.text_det_limit_type),
                paddlex_config_yaml: push_opt(&mut cstrings, &model.paddlex_config_yaml),

                textline_orientation_batch_size: model.textline_orientation_batch_size.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                text_recognition_batch_size: model.text_recognition_batch_size.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                use_doc_orientation_classify: model.use_doc_orientation_classify.map(|v| if v { 1 } else { 0 }).unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                use_doc_unwarping: model.use_doc_unwarping.map(|v| if v { 1 } else { 0 }).unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                use_textline_orientation: model.use_textline_orientation.map(|v| if v { 1 } else { 0 }).unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                text_det_limit_side_len: model.text_det_limit_side_len.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                text_det_max_side_limit: model.text_det_max_side_limit.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                enable_mkldnn: model.enable_mkldnn.map(|v| if v { 1 } else { 0 }).unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                mkldnn_cache_capacity: model.mkldnn_cache_capacity.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                cpu_threads: model.cpu_threads.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                thread_num: model.thread_num.unwrap_or(PADDLEOCR_OPTION_UNSET_I32),
                enable_struct_json: if pipeline.enable_struct_json { 1 } else { 0 },

                text_det_thresh: model.text_det_thresh.unwrap_or(f32::NAN),
                text_det_box_thresh: model.text_det_box_thresh.unwrap_or(f32::NAN),
                text_det_unclip_ratio: model.text_det_unclip_ratio.unwrap_or(f32::NAN),
                text_rec_score_thresh: model.text_rec_score_thresh.unwrap_or(f32::NAN),

                text_det_input_shape: [0; 4],
                text_rec_input_shape: [0; 4],
                text_det_input_shape_len: 0,
                text_rec_input_shape_len: 0,
            };

            if let Some(shape) = &model.text_det_input_shape {
                options.text_det_input_shape_len = shape.len().min(4) as i32;
                for (idx, value) in shape.iter().take(4).enumerate() {
                    options.text_det_input_shape[idx] = *value;
                }
            }
            if let Some(shape) = &model.text_rec_input_shape {
                options.text_rec_input_shape_len = shape.len().min(4) as i32;
                for (idx, value) in shape.iter().take(4).enumerate() {
                    options.text_rec_input_shape[idx] = *value;
                }
            }

            let mut handle: *mut c_void = std::ptr::null_mut();
            let code = unsafe { paddleocr_create(&options, &mut handle) };
            if code != 0 || handle.is_null() {
                set_last_error(&last_error_message());
                return Err(SdkErr::Runtime);
            }

            Ok(Self {
                handle,
                enable_struct_json: pipeline.enable_struct_json,
            })
        }
    }

    pub fn run_image(&mut self, data: *const u8, len: usize) -> SdkErr {
        #[cfg(not(feature = "paddleocr"))]
        {
            let _ = data;
            let _ = len;
            set_last_error("paddleocr feature not enabled");
            return SdkErr::Unsupported;
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            if data.is_null() || len == 0 {
                set_last_error("input data is null or empty");
                return SdkErr::InvalidArg;
            }
            let code = paddleocr_run_image(self.handle, data, len);
            if code != 0 {
                set_last_error(&last_error_message());
                return SdkErr::Runtime;
            }
            SdkErr::Ok
        }
    }

    pub fn line_count(&self) -> Result<usize, SdkErr> {
        #[cfg(not(feature = "paddleocr"))]
        {
            set_last_error("paddleocr feature not enabled");
            return Err(SdkErr::Unsupported);
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            let mut count = 0usize;
            let code = paddleocr_get_line_count(self.handle, &mut count);
            if code != 0 {
                set_last_error(&last_error_message());
                return Err(SdkErr::Runtime);
            }
            Ok(count)
        }
    }

    pub fn write_lines(&self, dst: *mut SdkOcrLine, cap: usize, out_written: *mut usize) -> SdkErr {
        #[cfg(not(feature = "paddleocr"))]
        {
            let _ = dst;
            let _ = cap;
            let _ = out_written;
            set_last_error("paddleocr feature not enabled");
            return SdkErr::Unsupported;
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let code = paddleocr_write_lines(self.handle, dst as *mut PaddleOcrLine, cap, out_written);
            if code != 0 {
                set_last_error(&last_error_message());
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }
    }

    pub fn text_bytes(&self) -> Result<usize, SdkErr> {
        #[cfg(not(feature = "paddleocr"))]
        {
            set_last_error("paddleocr feature not enabled");
            return Err(SdkErr::Unsupported);
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            let mut bytes = 0usize;
            let code = paddleocr_get_text_bytes(self.handle, &mut bytes);
            if code != 0 {
                set_last_error(&last_error_message());
                return Err(SdkErr::Runtime);
            }
            Ok(bytes)
        }
    }

    pub fn write_text(&self, dst: *mut i8, cap: usize, out_written: *mut usize) -> SdkErr {
        #[cfg(not(feature = "paddleocr"))]
        {
            let _ = dst;
            let _ = cap;
            let _ = out_written;
            set_last_error("paddleocr feature not enabled");
            return SdkErr::Unsupported;
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let code = paddleocr_write_text(self.handle, dst, cap, out_written);
            if code != 0 {
                set_last_error(&last_error_message());
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }
    }

    pub fn struct_json_bytes(&self) -> Result<usize, SdkErr> {
        if !self.enable_struct_json {
            return Ok(0);
        }

        #[cfg(not(feature = "paddleocr"))]
        {
            set_last_error("paddleocr feature not enabled");
            return Err(SdkErr::Unsupported);
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            let mut bytes = 0usize;
            let code = paddleocr_get_struct_json_bytes(self.handle, &mut bytes);
            if code != 0 {
                set_last_error(&last_error_message());
                return Err(SdkErr::Runtime);
            }
            Ok(bytes)
        }
    }

    pub fn write_struct_json(&self, dst: *mut i8, cap: usize, out_written: *mut usize) -> SdkErr {
        if !self.enable_struct_json {
            if !out_written.is_null() {
                unsafe { *out_written = 0; }
            }
            return SdkErr::Ok;
        }

        #[cfg(not(feature = "paddleocr"))]
        {
            let _ = dst;
            let _ = cap;
            let _ = out_written;
            set_last_error("paddleocr feature not enabled");
            return SdkErr::Unsupported;
        }

        #[cfg(feature = "paddleocr")]
        unsafe {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let code = paddleocr_write_struct_json(self.handle, dst, cap, out_written);
            if code != 0 {
                set_last_error(&last_error_message());
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }
    }
}

impl Drop for PaddleOcrPipeline {
    fn drop(&mut self) {
        #[cfg(feature = "paddleocr")]
        unsafe {
            let _ = paddleocr_destroy(self.handle);
        }
    }
}

#[cfg(feature = "paddleocr")]
fn last_error_message() -> String {
    unsafe {
        let mut len = 0usize;
        let ptr = paddleocr_last_error(&mut len);
        if ptr.is_null() || len == 0 {
            return "paddleocr error".to_string();
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, len);
        String::from_utf8_lossy(bytes).to_string()
    }
}

#[cfg(not(feature = "paddleocr"))]
fn last_error_message() -> String {
    "paddleocr not enabled".to_string()
}
