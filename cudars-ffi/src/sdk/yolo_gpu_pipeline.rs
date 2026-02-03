use libc::c_void;

use cudars_core::SdkErr;

use crate::CudaRsResult;
use crate::preprocess_gpu::CudaRsPreprocessResult;

use super::sdk_error::{handle_cudars_result, set_last_error};
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;
use super::yolo_model_config::YoloModelConfig;
use super::yolo_output_buffer::YoloOutputBuffer;
use super::yolo_pipeline_config::YoloPipelineConfig;

pub struct YoloGpuPipeline {
    stream: u64,
    event: u64,
    decoder: u64,
    preprocess: u64,
    trt: u64,
    input_device: *mut c_void,
    input_bytes: u64,
    input_bytes_per_sample: u64,
    input_batch: i32,
    outputs: Vec<YoloOutputBuffer>,
    last_meta: SdkYoloPreprocessMeta,
}

impl YoloGpuPipeline {
    pub fn new(model: &YoloModelConfig, pipeline: &YoloPipelineConfig) -> Result<Self, SdkErr> {
        let result = unsafe { crate::cudars_device_set(model.device_id) };
        if handle_cudars_result(result, "device set") != SdkErr::Ok {
            return Err(SdkErr::InvalidArg);
        }

        let mut stream = 0u64;
        let result = unsafe { crate::cudars_stream_create(&mut stream) };
        if handle_cudars_result(result, "stream create") != SdkErr::Ok {
            return Err(SdkErr::Runtime);
        }

        let mut event = 0u64;
        let result = unsafe { crate::cudars_event_create(&mut event) };
        if handle_cudars_result(result, "event create") != SdkErr::Ok {
            unsafe { crate::cudars_stream_destroy(stream) };
            return Err(SdkErr::Runtime);
        }

        let mut decoder = 0u64;
        let result = unsafe {
            crate::cudars_image_decoder_create(
                &mut decoder,
                pipeline.max_input_width,
                pipeline.max_input_height,
                model.input_channels,
            )
        };
        if handle_cudars_result(result, "image decoder create") != SdkErr::Ok {
            unsafe {
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Runtime);
        }

        let mut preprocess = 0u64;
        let result = unsafe {
            crate::cudars_preprocess_create(
                &mut preprocess,
                model.input_width,
                model.input_height,
                model.input_channels,
                pipeline.max_input_width,
                pipeline.max_input_height,
            )
        };
        if handle_cudars_result(result, "preprocess create") != SdkErr::Ok {
            unsafe {
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Runtime);
        }

        let mut trt = 0u64;
        let model_path = match std::ffi::CString::new(model.model_path.as_str()) {
            Ok(path) => path,
            Err(_) => {
                set_last_error("model_path contains invalid NUL byte");
                unsafe {
                    crate::cudars_preprocess_destroy(preprocess);
                    crate::cudars_image_decoder_destroy(decoder);
                    crate::cudars_event_destroy(event);
                    crate::cudars_stream_destroy(stream);
                }
                return Err(SdkErr::InvalidArg);
            }
        };
        let result = unsafe { crate::cudars_trt_load_engine(model_path.as_ptr(), model.device_id, &mut trt) };
        if handle_cudars_result(result, "trt load engine") != SdkErr::Ok {
            unsafe {
                crate::cudars_preprocess_destroy(preprocess);
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Runtime);
        }

        let mut input_ptr: *mut c_void = std::ptr::null_mut();
        let mut input_bytes = 0u64;
        let result = unsafe { crate::cudars_trt_get_input_device_ptr(trt, 0, &mut input_ptr, &mut input_bytes) };
        if handle_cudars_result(result, "trt get input device ptr") != SdkErr::Ok {
            unsafe {
                crate::cudars_trt_destroy(trt);
                crate::cudars_preprocess_destroy(preprocess);
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Runtime);
        }

        let input_shape = get_trt_input_shape(trt);
        let input_batch = input_shape.first().copied().unwrap_or(1) as i32;
        if input_batch != 1 {
            set_last_error("only batch size 1 is supported in YoloGpuPipeline");
            unsafe {
                crate::cudars_trt_destroy(trt);
                crate::cudars_preprocess_destroy(preprocess);
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Unsupported);
        }

        if input_bytes % input_batch as u64 != 0 {
            set_last_error("input bytes not divisible by batch size");
            unsafe {
                crate::cudars_trt_destroy(trt);
                crate::cudars_preprocess_destroy(preprocess);
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::BadState);
        }

        let input_bytes_per_sample = input_bytes / input_batch as u64;

        let mut output_count = 0i32;
        let result = unsafe { crate::cudars_trt_get_output_count(trt, &mut output_count) };
        if handle_cudars_result(result, "trt get output count") != SdkErr::Ok {
            unsafe {
                crate::cudars_trt_destroy(trt);
                crate::cudars_preprocess_destroy(preprocess);
                crate::cudars_image_decoder_destroy(decoder);
                crate::cudars_event_destroy(event);
                crate::cudars_stream_destroy(stream);
            }
            return Err(SdkErr::Runtime);
        }

        let mut outputs = Vec::with_capacity(output_count as usize);
        for idx in 0..output_count {
            let mut out_ptr: *mut c_void = std::ptr::null_mut();
            let mut out_bytes = 0u64;
            let result = unsafe { crate::cudars_trt_get_output_device_ptr(trt, idx, &mut out_ptr, &mut out_bytes) };
            if handle_cudars_result(result, "trt get output device ptr") != SdkErr::Ok {
                cleanup_outputs(&mut outputs);
                unsafe {
                    crate::cudars_trt_destroy(trt);
                    crate::cudars_preprocess_destroy(preprocess);
                    crate::cudars_image_decoder_destroy(decoder);
                    crate::cudars_event_destroy(event);
                    crate::cudars_stream_destroy(stream);
                }
                return Err(SdkErr::Runtime);
            }

            let shape = get_trt_output_shape(trt, idx);
            let mut host_pinned: *mut c_void = std::ptr::null_mut();
            let result = unsafe { crate::cudars_host_alloc_pinned(&mut host_pinned, out_bytes as usize) };
            if handle_cudars_result(result, "host alloc pinned") != SdkErr::Ok {
                cleanup_outputs(&mut outputs);
                unsafe {
                    crate::cudars_trt_destroy(trt);
                    crate::cudars_preprocess_destroy(preprocess);
                    crate::cudars_image_decoder_destroy(decoder);
                    crate::cudars_event_destroy(event);
                    crate::cudars_stream_destroy(stream);
                }
                return Err(SdkErr::OutOfMemory);
            }

            outputs.push(YoloOutputBuffer {
                device_ptr: out_ptr,
                host_pinned,
                bytes: out_bytes,
                shape,
            });
        }

        Ok(Self {
            stream,
            event,
            decoder,
            preprocess,
            trt,
            input_device: input_ptr,
            input_bytes,
            input_bytes_per_sample,
            input_batch,
            outputs,
            last_meta: SdkYoloPreprocessMeta::default(),
        })
    }

    pub fn run_image(&mut self, data: *const u8, len: usize, meta: *mut SdkYoloPreprocessMeta) -> SdkErr {
        if data.is_null() || len == 0 {
            set_last_error("input data is null or empty");
            return SdkErr::InvalidArg;
        }

        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let mut pitch = 0i32;
        let mut width = 0i32;
        let mut height = 0i32;
        let mut format = 0i32;

        let result = unsafe {
            crate::cudars_image_decoder_decode_to_device(
                self.decoder,
                data as *const u8,
                len,
                self.stream,
                &mut dev_ptr,
                &mut pitch,
                &mut width,
                &mut height,
                &mut format,
            )
        };
        if handle_cudars_result(result, "image decode") != SdkErr::Ok {
            return SdkErr::Runtime;
        }

        let mut prep = CudaRsPreprocessResult {
            output_ptr: std::ptr::null_mut(),
            output_size: 0,
            scale: 0.0,
            pad_x: 0,
            pad_y: 0,
            original_width: 0,
            original_height: 0,
        };

        let result = unsafe {
            crate::cudars_preprocess_run_device_on_stream_into(
                self.preprocess,
                dev_ptr as *const u8,
                width,
                height,
                self.stream,
                0,
                self.input_device as *mut f32,
                &mut prep,
            )
        };
        if handle_cudars_result(result, "preprocess") != SdkErr::Ok {
            return SdkErr::Runtime;
        }

        let input_len = self.input_bytes / std::mem::size_of::<f32>() as u64;
        let result = unsafe {
            crate::cudars_trt_enqueue_device(
                self.trt,
                self.input_device as *const f32,
                input_len,
                self.stream,
                0,
            )
        };
        if handle_cudars_result(result, "trt enqueue") != SdkErr::Ok {
            return SdkErr::Runtime;
        }

        for output in &self.outputs {
            let result = unsafe {
                crate::cudars_memcpy_dtoh_async_raw(
                    output.host_pinned,
                    output.device_ptr,
                    output.bytes as usize,
                    self.stream,
                )
            };
            if handle_cudars_result(result, "dtoh async") != SdkErr::Ok {
                return SdkErr::Runtime;
            }
        }

        let result = unsafe { crate::cudars_event_record(self.event, self.stream) };
        if handle_cudars_result(result, "event record") != SdkErr::Ok {
            return SdkErr::Runtime;
        }

        let result = unsafe { crate::cudars_event_synchronize(self.event) };
        if handle_cudars_result(result, "event synchronize") != SdkErr::Ok {
            return SdkErr::Runtime;
        }

        self.last_meta = SdkYoloPreprocessMeta {
            scale: prep.scale,
            pad_x: prep.pad_x,
            pad_y: prep.pad_y,
            original_width: prep.original_width,
            original_height: prep.original_height,
        };

        if !meta.is_null() {
            unsafe {
                *meta = self.last_meta;
            }
        }

        SdkErr::Ok
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub fn output_shape(&self, index: usize) -> Option<&[i64]> {
        self.outputs.get(index).map(|o| o.shape.as_slice())
    }

    pub fn output_bytes(&self, index: usize) -> Option<usize> {
        self.outputs.get(index).map(|o| o.bytes as usize)
    }

    pub fn read_output(&self, index: usize, dst: *mut u8, cap: usize, out_written: *mut usize) -> SdkErr {
        if dst.is_null() || out_written.is_null() {
            set_last_error("dst or out_written is null");
            return SdkErr::InvalidArg;
        }

        let output = match self.outputs.get(index) {
            Some(o) => o,
            None => {
                set_last_error("output index out of range");
                return SdkErr::InvalidArg;
            }
        };

        let bytes = output.bytes as usize;
        let to_copy = bytes.min(cap);
        unsafe {
            std::ptr::copy_nonoverlapping(output.host_pinned as *const u8, dst, to_copy);
            *out_written = to_copy;
        }

        if to_copy < bytes {
            set_last_error("destination buffer too small");
            return SdkErr::InvalidArg;
        }

        SdkErr::Ok
    }
}

impl Drop for YoloGpuPipeline {
    fn drop(&mut self) {
        for output in &self.outputs {
            unsafe {
                let _ = crate::cudars_host_free_pinned(output.host_pinned);
            }
        }
        unsafe {
            let _ = crate::cudars_trt_destroy(self.trt);
            let _ = crate::cudars_preprocess_destroy(self.preprocess);
            let _ = crate::cudars_image_decoder_destroy(self.decoder);
            let _ = crate::cudars_event_destroy(self.event);
            let _ = crate::cudars_stream_destroy(self.stream);
        }
    }
}

fn cleanup_outputs(outputs: &mut Vec<YoloOutputBuffer>) {
    for output in outputs.iter() {
        unsafe {
            let _ = crate::cudars_host_free_pinned(output.host_pinned);
        }
    }
    outputs.clear();
}

fn get_trt_input_shape(trt: u64) -> Vec<i64> {
    let mut shape = [0i64; 16];
    let mut shape_len = 0i32;
    let result = unsafe { crate::cudars_trt_get_input_info(trt, 0, shape.as_mut_ptr(), &mut shape_len, 16) };
    if result != CudaRsResult::Success || shape_len <= 0 {
        return Vec::new();
    }
    shape[..shape_len as usize].to_vec()
}

fn get_trt_output_shape(trt: u64, index: i32) -> Vec<i64> {
    let mut shape = [0i64; 16];
    let mut shape_len = 0i32;
    let result = unsafe { crate::cudars_trt_get_output_info(trt, index, shape.as_mut_ptr(), &mut shape_len, 16) };
    if result != CudaRsResult::Success || shape_len <= 0 {
        return Vec::new();
    }
    shape[..shape_len as usize].to_vec()
}
