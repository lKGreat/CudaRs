use cudars_core::SdkErr;

#[cfg(feature = "openvino")]
use super::openvino_output::OpenVinoOutput;
use super::sdk_error::{handle_cudars_result, set_last_error};
use super::openvino_model_config::OpenVinoModelConfig;
use super::openvino_pipeline_config::OpenVinoPipelineConfig;

#[cfg(feature = "openvino")]
mod imp {
    use super::*;
    use crate::{
        cudars_ov_destroy, cudars_ov_free_tensors, cudars_ov_load, cudars_ov_run, CudaRsOvConfig,
        CudaRsOvDevice, CudaRsOvModel, CudaRsOvTensor,
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

            let (device, device_index) = parse_openvino_device(&pipeline.openvino_device)?;
            let mut config = CudaRsOvConfig {
                device,
                device_index,
                num_streams: 0,
                enable_profiling: 0,
                properties_json_ptr: ptr::null(),
                properties_json_len: 0,
            };

            let mut props_cstr: Option<CString> = None;
            if !pipeline.openvino_config_json.trim().is_empty() {
                let cstr = CString::new(pipeline.openvino_config_json.as_str())
                    .map_err(|_| SdkErr::InvalidArg)?;
                config.properties_json_ptr = cstr.as_ptr();
                config.properties_json_len = pipeline.openvino_config_json.len();
                props_cstr = Some(cstr);
            }

            let mut handle: CudaRsOvModel = 0;
            let model_path = CString::new(model.model_path.as_str()).map_err(|_| SdkErr::InvalidArg)?;
            let result = cudars_ov_load(model_path.as_ptr(), &config, &mut handle);
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

    fn parse_openvino_device(device: &str) -> Result<(CudaRsOvDevice, c_int), SdkErr> {
        let d = device.trim().to_lowercase();
        if d.is_empty() || d == "auto" {
            return Ok((CudaRsOvDevice::Auto, 0));
        }
        if d == "cpu" {
            return Ok((CudaRsOvDevice::Cpu, 0));
        }
        if d == "gpu" {
            return Ok((CudaRsOvDevice::Gpu, 0));
        }
        if d.starts_with("gpu:") || d.starts_with("gpu.") {
            let idx = d[4..].parse::<c_int>().map_err(|_| SdkErr::InvalidArg)?;
            return Ok((CudaRsOvDevice::GpuIndex, idx));
        }
        if d == "npu" {
            return Ok((CudaRsOvDevice::Npu, 0));
        }

        set_last_error("invalid openvino device");
        Err(SdkErr::InvalidArg)
    }
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
    }
}

pub use imp::OpenVinoTensorPipeline;
