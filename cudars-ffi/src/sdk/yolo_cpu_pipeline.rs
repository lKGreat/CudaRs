use cudars_core::SdkErr;

use super::sdk_error::set_last_error;
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;
#[cfg(feature = "onnxruntime")]
use super::yolo_preprocess_cpu::{decode_image_rgb, letterbox_u8_to_tensor, InputLayout};
use super::yolo_model_config::YoloModelConfig;
use super::yolo_pipeline_config::YoloPipelineConfig;

#[cfg(feature = "onnxruntime")]
mod imp {
    use super::*;
    use ndarray::IxDyn;
    use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel, LoggingLevel};

    struct OrtSession(Session<'static>);
    unsafe impl Send for OrtSession {}

    struct YoloCpuOutput {
        shape: Vec<i64>,
        data: Vec<f32>,
    }

    pub struct YoloCpuPipeline {
        session: OrtSession,
        input_width: i32,
        input_height: i32,
        input_channels: i32,
        input_layout: InputLayout,
        outputs: Vec<YoloCpuOutput>,
        last_meta: SdkYoloPreprocessMeta,
    }

    unsafe impl Send for YoloCpuPipeline {}

    lazy_static::lazy_static! {
        static ref ORT_ENV: Environment = Environment::builder()
            .with_name("cudars_ort_cpu")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .expect("Failed to create ONNX Runtime environment");
    }

    impl YoloCpuPipeline {
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
                set_last_error("only 3-channel input supported for CPU pipeline");
                return Err(SdkErr::Unsupported);
            }

            let session = make_session(&model.model_path, pipeline.cpu_threads)?;
            let input_layout = infer_input_layout(&session, model.input_channels);
            if std::env::var("CUDARS_DIAG").as_deref() == Ok("1") {
                if let Some(input) = session.inputs.first() {
                    eprintln!(
                        "[cudars] ort input dims={:?} layout={:?}",
                        input.dimensions, input_layout
                    );
                }
            }

            Ok(Self {
                session: OrtSession(session),
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

            let profile = std::env::var("CUDARS_PROFILE").as_deref() == Ok("1");
            let t0 = if profile { Some(std::time::Instant::now()) } else { None };
            let bytes = unsafe { std::slice::from_raw_parts(data, len) };
            let (rgb, width, height) = match decode_image_rgb(bytes) {
                Ok(v) => v,
                Err(err) => return err,
            };
            let t_decode = t0.map(|t| t.elapsed());

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
            let t_preprocess = t0.map(|t| t.elapsed());

            let shape = match self.input_layout {
                InputLayout::Nchw => [
                    1usize,
                    self.input_channels as usize,
                    self.input_height as usize,
                    self.input_width as usize,
                ],
                InputLayout::Nhwc => [
                    1usize,
                    self.input_height as usize,
                    self.input_width as usize,
                    self.input_channels as usize,
                ],
            };
            let input_tensor = match ndarray::Array::from_shape_vec(IxDyn(&shape), input) {
                Ok(t) => t,
                Err(_) => {
                    set_last_error("failed to build input tensor");
                    return SdkErr::InvalidArg;
                }
            };

            let outputs: Vec<onnxruntime::tensor::OrtOwnedTensor<f32, IxDyn>> =
                match self.session.0.run(vec![input_tensor]) {
                    Ok(o) => o,
                    Err(err) => {
                        set_last_error(&format!("onnxruntime run failed: {err}"));
                        return SdkErr::Runtime;
                    }
                };
            let t_run = t0.map(|t| t.elapsed());

            self.outputs.clear();
            self.outputs.reserve(outputs.len());
            for output in outputs {
                let shape_vec: Vec<i64> = output.shape().iter().map(|d| *d as i64).collect();
                let data_vec: Vec<f32> = output.iter().copied().collect();
                self.outputs.push(YoloCpuOutput {
                    shape: shape_vec,
                    data: data_vec,
                });
            }
            let t_outputs = t0.map(|t| t.elapsed());

            self.last_meta = preprocess;
            if !meta.is_null() {
                unsafe {
                    *meta = preprocess;
                }
            }

            if profile {
                let decode_ms = t_decode.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
                let preprocess_ms = t_preprocess.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
                let run_ms = t_run.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
                let outputs_ms = t_outputs.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
                eprintln!(
                    "[cudars][cpu] decode={decode_ms:.2}ms preprocess={preprocess_ms:.2}ms ort_run={run_ms:.2}ms outputs={outputs_ms:.2}ms"
                );
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
            self.outputs.get(index).map(|o| o.data.len() * std::mem::size_of::<f32>())
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

            let bytes = output.data.len() * std::mem::size_of::<f32>();
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

    fn make_session(model_path: &str, cpu_threads: i32) -> Result<Session<'static>, SdkErr> {
        let threads = if cpu_threads > 0 { cpu_threads } else { 1 };
        let threads_i16 = if threads > i16::MAX as i32 {
            i16::MAX
        } else {
            threads as i16
        };
        let session = ORT_ENV
            .new_session_builder()
            .map_err(|err| {
                set_last_error(&format!("failed to create ONNX Runtime session builder: {err}"));
                SdkErr::Runtime
            })?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|err| {
                set_last_error(&format!("failed to set ONNX Runtime optimization level: {err}"));
                SdkErr::Runtime
            })?
            .with_number_threads(threads_i16)
            .map_err(|err| {
                set_last_error(&format!("failed to set ONNX Runtime thread count: {err}"));
                SdkErr::Runtime
            })?
            .with_model_from_file(model_path)
            .map_err(|err| {
                set_last_error(&format!("failed to load ONNX model: {err}"));
                SdkErr::Runtime
            })?;

        let session_static: Session<'static> = unsafe { std::mem::transmute::<Session<'_>, Session<'static>>(session) };
        Ok(session_static)
    }

    fn infer_input_layout(session: &Session<'static>, channels: i32) -> InputLayout {
        let ch = channels as u32;
        let input = match session.inputs.first() {
            Some(i) => i,
            None => return InputLayout::Nchw,
        };

        if input.dimensions.len() == 4 {
            if input.dimensions.get(1).and_then(|d| *d) == Some(ch) {
                return InputLayout::Nchw;
            }
            if input.dimensions.get(3).and_then(|d| *d) == Some(ch) {
                return InputLayout::Nhwc;
            }
        }

        InputLayout::Nchw
    }
}

#[cfg(not(feature = "onnxruntime"))]
mod imp {
    use super::*;

    pub struct YoloCpuPipeline;

    impl YoloCpuPipeline {
        pub fn new(_model: &YoloModelConfig, _pipeline: &YoloPipelineConfig) -> Result<Self, SdkErr> {
            set_last_error("onnxruntime feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn run_image(&mut self, _data: *const u8, _len: usize, _meta: *mut SdkYoloPreprocessMeta) -> SdkErr {
            set_last_error("onnxruntime feature not enabled");
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
            set_last_error("onnxruntime feature not enabled");
            SdkErr::Unsupported
        }
    }
}

pub use imp::YoloCpuPipeline;
