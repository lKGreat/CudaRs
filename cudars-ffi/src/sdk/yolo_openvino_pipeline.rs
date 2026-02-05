use cudars_core::SdkErr;

use super::sdk_error::{handle_cudars_result, set_last_error};
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;
#[cfg(feature = "openvino")]
use super::openvino_output::OpenVinoOutput;
use super::yolo_model_config::YoloModelConfig;
use super::yolo_pipeline_config::YoloPipelineConfig;
use super::openvino_config_utils::{build_openvino_properties_json, parse_openvino_device};
#[cfg(feature = "openvino")]
use super::yolo_preprocess_cpu::{decode_image_rgb, letterbox_u8_to_tensor, InputLayout};

#[cfg(feature = "openvino")]
mod imp {
    use super::*;
    use crate::{
        cudars_ov_destroy, cudars_ov_free_tensors, cudars_ov_get_input_info, cudars_ov_load_v2,
        cudars_ov_run, CudaRsOvConfigV2, CudaRsOvModel, CudaRsOvTensor, CudaRsResult,
    };
    use std::ffi::CString;
    use std::ptr;

    pub struct YoloOpenVinoPipeline {
        model: CudaRsOvModel,
        input_width: i32,
        input_height: i32,
        input_channels: i32,
        input_layout: InputLayout,
        outputs: Vec<OpenVinoOutput>,
        last_meta: SdkYoloPreprocessMeta,
    }

    unsafe impl Send for YoloOpenVinoPipeline {}

    impl YoloOpenVinoPipeline {
        pub fn new(model: &YoloModelConfig, pipeline: &YoloPipelineConfig) -> Result<Self, SdkErr> {
            if model.model_path.is_empty() {
                set_last_error("model_path is required");
                return Err(SdkErr::InvalidArg);
            }
            if model.input_width <= 0 || model.input_height <= 0 || model.input_channels <= 0 {
                set_last_error("invalid input dimensions");
                return Err(SdkErr::InvalidArg);
            }
            if model.input_channels != 3 {
                set_last_error("only 3-channel input supported for OpenVINO pipeline");
                return Err(SdkErr::Unsupported);
            }

            let device_spec = parse_openvino_device(&pipeline.openvino_device)?;
            let mut config = CudaRsOvConfigV2 {
                struct_size: std::mem::size_of::<CudaRsOvConfigV2>() as u32,
                device: device_spec.device,
                device_index: device_spec.device_index,
                device_name_ptr: ptr::null(),
                device_name_len: 0,
                num_streams: pipeline.openvino_num_streams,
                enable_profiling: if pipeline.openvino_enable_profiling { 1 } else { 0 },
                properties_json_ptr: ptr::null(),
                properties_json_len: 0,
            };

            let num_requests = if pipeline.openvino_num_requests > 0 {
                pipeline.openvino_num_requests
            } else {
                pipeline.worker_count.max(1)
            };

            let perf_mode = if !pipeline.openvino_performance_mode.trim().is_empty() {
                pipeline.openvino_performance_mode.as_str()
            } else if num_requests > 1 || pipeline.worker_count > 1 {
                "throughput"
            } else {
                ""
            };

            let properties_json = build_openvino_properties_json(
                pipeline.openvino_config_json.as_str(),
                perf_mode,
                num_requests,
                pipeline.openvino_cache_dir.as_str(),
                pipeline.openvino_enable_mmap,
            )?;

            let mut props_cstr: Option<CString> = None;
            if let Some(json) = properties_json {
                let cstr = CString::new(json.as_str()).map_err(|_| SdkErr::InvalidArg)?;
                config.properties_json_ptr = cstr.as_ptr();
                config.properties_json_len = json.len();
                props_cstr = Some(cstr);
            }

            let mut device_name_cstr: Option<CString> = None;
            if let Some(name) = device_spec.device_name_override {
                let cstr = CString::new(name.as_str()).map_err(|_| SdkErr::InvalidArg)?;
                config.device_name_ptr = cstr.as_ptr();
                config.device_name_len = name.len();
                device_name_cstr = Some(cstr);
            }

            let mut handle: CudaRsOvModel = 0;
            let model_path = CString::new(model.model_path.as_str()).map_err(|_| SdkErr::InvalidArg)?;
            let result = cudars_ov_load_v2(model_path.as_ptr(), &config, &mut handle);
            drop(device_name_cstr);
            drop(props_cstr);
            let err = handle_cudars_result(result, "openvino load");
            if err != SdkErr::Ok {
                return Err(err);
            }

            let input_layout = query_input_layout(handle, model.input_channels);
            if std::env::var("CUDARS_OV_DEBUG").as_deref() == Ok("1") {
                eprintln!("[cudars][openvino] inferred input layout: {:?}", input_layout);
            }

            Ok(Self {
                model: handle,
                input_width: model.input_width,
                input_height: model.input_height,
                input_channels: model.input_channels,
                input_layout,
                outputs: Vec::new(),
                last_meta: SdkYoloPreprocessMeta::default(),
            })
        }

        pub fn run_image(&mut self, data: *const u8, len: usize, meta: *mut SdkYoloPreprocessMeta) -> SdkErr {
            if data.is_null() || len == 0 {
                set_last_error("input data is null or empty");
                return SdkErr::InvalidArg;
            }

            let bytes = unsafe { std::slice::from_raw_parts(data, len) };
            let (rgb, width, height) = match decode_image_rgb(bytes) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let (input, preprocess) = match letterbox_u8_to_tensor(
                &rgb,
                width,
                height,
                self.input_width,
                self.input_height,
                self.input_channels,
                self.input_layout,
            ) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let shape: Vec<i64> = match self.input_layout {
                InputLayout::Nchw => vec![
                    1,
                    self.input_channels as i64,
                    self.input_height as i64,
                    self.input_width as i64,
                ],
                InputLayout::Nhwc => vec![
                    1,
                    self.input_height as i64,
                    self.input_width as i64,
                    self.input_channels as i64,
                ],
            };

            let mut out_tensors: *mut CudaRsOvTensor = ptr::null_mut();
            let mut out_count: u64 = 0;
            let result = cudars_ov_run(
                self.model,
                input.as_ptr(),
                input.len() as u64,
                shape.as_ptr(),
                shape.len() as u64,
                &mut out_tensors,
                &mut out_count,
            );
            let err = handle_cudars_result(result, "openvino run");
            if err != SdkErr::Ok {
                return err;
            }

            unsafe {
                let slice = std::slice::from_raw_parts(out_tensors, out_count as usize);
                self.outputs.clear();
                self.outputs.reserve(slice.len());
                for t in slice {
                    let data = std::slice::from_raw_parts(t.data, t.data_len as usize).to_vec();
                    let shape = std::slice::from_raw_parts(t.shape, t.shape_len as usize).to_vec();
                    self.outputs.push(OpenVinoOutput { shape, data });
                }
                let _ = cudars_ov_free_tensors(out_tensors, out_count as u64);
            }

            self.last_meta = preprocess;
            if !meta.is_null() {
                unsafe { *meta = preprocess; }
            }

            SdkErr::Ok
        }

        pub fn run_batch_images(
            &mut self,
            images: *const *const u8,
            image_lens: *const usize,
            batch_size: usize,
            out_metas: *mut SdkYoloPreprocessMeta,
        ) -> SdkErr {
            if images.is_null() || image_lens.is_null() || batch_size == 0 {
                set_last_error("invalid batch parameters");
                return SdkErr::InvalidArg;
            }

            // Decode and preprocess all images
            let images_slice = unsafe { std::slice::from_raw_parts(images, batch_size) };
            let lens_slice = unsafe { std::slice::from_raw_parts(image_lens, batch_size) };

            let mut preprocessed: Vec<(Vec<f32>, SdkYoloPreprocessMeta)> = Vec::with_capacity(batch_size);
            
            for i in 0..batch_size {
                let data = images_slice[i];
                let len = lens_slice[i];
                
                if data.is_null() || len == 0 {
                    set_last_error("image data is null or empty in batch");
                    return SdkErr::InvalidArg;
                }

                let bytes = unsafe { std::slice::from_raw_parts(data, len) };
                let (rgb, width, height) = match decode_image_rgb(bytes) {
                    Ok(v) => v,
                    Err(err) => return err,
                };

                let (input, meta) = match letterbox_u8_to_tensor(
                    &rgb,
                    width,
                    height,
                    self.input_width,
                    self.input_height,
                    self.input_channels,
                    self.input_layout,
                ) {
                    Ok(v) => v,
                    Err(err) => return err,
                };

                preprocessed.push((input, meta));
            }

            // Prepare batch input data
            let single_size = self.input_channels as usize * self.input_height as usize * self.input_width as usize;
            let mut batch_inputs: Vec<*const f32> = Vec::with_capacity(batch_size);
            let mut batch_lens: Vec<u64> = Vec::with_capacity(batch_size);
            
            for (input, _) in &preprocessed {
                batch_inputs.push(input.as_ptr());
                batch_lens.push(input.len() as u64);
            }

            // Create single shape (without batch dimension)
            let single_shape: Vec<i64> = match self.input_layout {
                InputLayout::Nchw => vec![
                    self.input_channels as i64,
                    self.input_height as i64,
                    self.input_width as i64,
                ],
                InputLayout::Nhwc => vec![
                    self.input_height as i64,
                    self.input_width as i64,
                    self.input_channels as i64,
                ],
            };

            // Call batch inference
            let mut out_batch_tensors: *mut *mut CudaRsOvTensor = ptr::null_mut();
            let mut out_batch_counts: *mut u64 = ptr::null_mut();
            
            let result = crate::cudars_ov_run_batch(
                self.model,
                batch_inputs.as_ptr(),
                batch_lens.as_ptr(),
                batch_size as u64,
                single_shape.as_ptr(),
                single_shape.len() as u64,
                &mut out_batch_tensors,
                &mut out_batch_counts,
            );

            let err = handle_cudars_result(result, "openvino run batch");
            if err != SdkErr::Ok {
                return err;
            }

            // For now, only support returning the first image's outputs
            // In a full implementation, would need to handle all batch outputs
            unsafe {
                if out_batch_tensors.is_null() || out_batch_counts.is_null() {
                    set_last_error("batch output is null");
                    return SdkErr::Unknown;
                }

                // Get first image's outputs
                let first_tensors = *out_batch_tensors;
                let first_count = *out_batch_counts;

                let slice = std::slice::from_raw_parts(first_tensors, first_count as usize);
                self.outputs.clear();
                self.outputs.reserve(slice.len());
                for t in slice {
                    let data = std::slice::from_raw_parts(t.data, t.data_len as usize).to_vec();
                    let shape = std::slice::from_raw_parts(t.shape, t.shape_len as usize).to_vec();
                    self.outputs.push(OpenVinoOutput { shape, data });
                }

                // Store first image's meta
                if !preprocessed.is_empty() {
                    self.last_meta = preprocessed[0].1;
                    if !out_metas.is_null() {
                        let metas_slice = std::slice::from_raw_parts_mut(out_metas, batch_size);
                        for i in 0..batch_size {
                            metas_slice[i] = preprocessed[i].1;
                        }
                    }
                }

                // Free batch tensors
                let _ = crate::cudars_ov_free_batch_tensors(out_batch_tensors, out_batch_counts, batch_size as u64);
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
            self.outputs.get(index).map(|o| o.bytes())
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

            let bytes = output.bytes();
            let to_copy = bytes.min(cap);
            unsafe {
                std::ptr::copy_nonoverlapping(output.data.as_ptr() as *const u8, dst, to_copy);
                *out_written = to_copy;
            }

            if to_copy < bytes {
                set_last_error("destination buffer too small");
                return SdkErr::InvalidArg;
            }

            SdkErr::Ok
        }
    }

    impl Drop for YoloOpenVinoPipeline {
        fn drop(&mut self) {
            let _ = cudars_ov_destroy(self.model);
        }
    }

    // device parsing handled by openvino_config_utils

    fn query_input_layout(handle: CudaRsOvModel, channels: i32) -> InputLayout {
        let mut shape = [0i64; 8];
        let mut shape_len: i32 = 0;
        let result = cudars_ov_get_input_info(handle, 0, shape.as_mut_ptr(), &mut shape_len, 8);
        if result != CudaRsResult::Success || shape_len <= 0 {
            return InputLayout::Nchw;
        }

        let dims = &shape[..shape_len as usize];
        infer_layout_from_shape(dims, channels)
    }

    fn infer_layout_from_shape(shape: &[i64], channels: i32) -> InputLayout {
        if shape.len() == 4 {
            if shape.get(1).copied().unwrap_or(0) == channels as i64 {
                return InputLayout::Nchw;
            }
            if shape.get(3).copied().unwrap_or(0) == channels as i64 {
                return InputLayout::Nhwc;
            }
        }
        InputLayout::Nchw
    }
}

#[cfg(not(feature = "openvino"))]
mod imp {
    use super::*;

    pub struct YoloOpenVinoPipeline;

    impl YoloOpenVinoPipeline {
        pub fn new(_model: &YoloModelConfig, _pipeline: &YoloPipelineConfig) -> Result<Self, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn run_image(&mut self, _data: *const u8, _len: usize, _meta: *mut SdkYoloPreprocessMeta) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn run_batch_images(
            &mut self,
            _images: *const *const u8,
            _image_lens: *const usize,
            _batch_size: usize,
            _out_metas: *mut SdkYoloPreprocessMeta,
        ) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn output_count(&self) -> usize {
            0
        }

        pub fn output_shape(&self, _index: usize) -> Option<&[i64]> {
            None
        }

        pub fn output_bytes(&self, _index: usize) -> Option<usize> {
            None
        }

        pub fn read_output(&self, _index: usize, _dst: *mut u8, _cap: usize, _out_written: *mut usize) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }
    }
}

pub use imp::YoloOpenVinoPipeline;
