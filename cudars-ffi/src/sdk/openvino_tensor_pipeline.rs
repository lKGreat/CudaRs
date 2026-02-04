use cudars_core::SdkErr;

#[cfg(feature = "openvino")]
use super::openvino_output::OpenVinoOutput;
use super::sdk_error::{handle_cudars_result, set_last_error};
use super::openvino_model_config::OpenVinoModelConfig;
use super::openvino_pipeline_config::OpenVinoPipelineConfig;
use super::openvino_config_utils::{build_openvino_properties_json, parse_openvino_device};

#[cfg(feature = "openvino")]
mod imp {
    use super::*;
    use crate::{
        cudars_ov_destroy, cudars_ov_free_tensors, cudars_ov_load_v2, cudars_ov_run,
        cudars_ov_async_queue_submit, cudars_ov_async_queue_wait,
        CudaRsOvConfigV2, CudaRsOvModel, CudaRsOvTensor,
    };
    use libc::c_int;
    use std::ffi::CString;
    use std::ptr;

    pub struct OpenVinoTensorPipeline {
        model: CudaRsOvModel,
        outputs: Vec<OpenVinoOutput>,
    }

    unsafe impl Send for OpenVinoTensorPipeline {}

    impl OpenVinoTensorPipeline {
        pub fn new(model: &OpenVinoModelConfig, pipeline: &OpenVinoPipelineConfig) -> Result<Self, SdkErr> {
            if model.model_path.is_empty() {
                set_last_error("model_path is required");
                return Err(SdkErr::InvalidArg);
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

            let perf_mode = if !pipeline.openvino_performance_mode.trim().is_empty() {
                pipeline.openvino_performance_mode.as_str()
            } else if pipeline.openvino_num_requests > 1 {
                "throughput"
            } else {
                ""
            };

            let properties_json = build_openvino_properties_json(
                pipeline.openvino_config_json.as_str(),
                perf_mode,
                pipeline.openvino_num_requests,
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

            Ok(Self {
                model: handle,
                outputs: Vec::new(),
            })
        }

        pub fn run_tensor(&mut self, input: *const f32, input_len: usize, shape: *const i64, shape_len: usize) -> SdkErr {
            if input.is_null() || shape.is_null() || input_len == 0 || shape_len == 0 {
                set_last_error("input or shape is null/empty");
                return SdkErr::InvalidArg;
            }

            let mut out_tensors: *mut CudaRsOvTensor = ptr::null_mut();
            let mut out_count: u64 = 0;
            let result = cudars_ov_run(
                self.model,
                input,
                input_len as u64,
                shape,
                shape_len as u64,
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

            SdkErr::Ok
        }

        pub fn submit_async(
            &mut self,
            input: *const f32,
            input_len: usize,
            shape: *const i64,
            shape_len: usize,
            out_request_id: *mut c_int,
        ) -> SdkErr {
            if input.is_null() || shape.is_null() || out_request_id.is_null() || input_len == 0 || shape_len == 0 {
                set_last_error("input/shape/request_id is null/empty");
                return SdkErr::InvalidArg;
            }

            let result = cudars_ov_async_queue_submit(
                self.model,
                input,
                input_len as u64,
                shape,
                shape_len as u64,
                out_request_id,
            );
            handle_cudars_result(result, "openvino async submit")
        }

        pub fn wait_async(&mut self, request_id: c_int) -> SdkErr {
            let mut out_tensors: *mut CudaRsOvTensor = ptr::null_mut();
            let mut out_count: u64 = 0;
            let result = cudars_ov_async_queue_wait(self.model, request_id, &mut out_tensors, &mut out_count);
            let err = handle_cudars_result(result, "openvino async wait");
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

    impl Drop for OpenVinoTensorPipeline {
        fn drop(&mut self) {
            let _ = cudars_ov_destroy(self.model);
        }
    }

    // device parsing handled by openvino_config_utils
}

#[cfg(not(feature = "openvino"))]
mod imp {
    use super::*;

    pub struct OpenVinoTensorPipeline;

    impl OpenVinoTensorPipeline {
        pub fn new(_model: &OpenVinoModelConfig, _pipeline: &OpenVinoPipelineConfig) -> Result<Self, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn run_tensor(&mut self, _input: *const f32, _input_len: usize, _shape: *const i64, _shape_len: usize) -> SdkErr {
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

        pub fn submit_async(
            &mut self,
            _input: *const f32,
            _input_len: usize,
            _shape: *const i64,
            _shape_len: usize,
            _out_request_id: *mut i32,
        ) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn wait_async(&mut self, _request_id: i32) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }
    }
}

pub use imp::OpenVinoTensorPipeline;
