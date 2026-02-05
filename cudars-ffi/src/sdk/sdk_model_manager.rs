use cudars_core::{ModelKind, PipelineKind, SdkErr};

use super::model_instance::ModelInstance;
use super::model_manager_state::ModelManagerState;
use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err};
use super::sdk_handles::{SdkModelHandle, SdkModelManagerHandle, SdkPipelineHandle, SDK_MODEL_MANAGERS, SDK_MODELS, SDK_PIPELINES};
use super::sdk_model_spec::SdkModelSpec;
use super::sdk_pipeline_spec::SdkPipelineSpec;
use super::sdk_strings::read_utf8;
use super::yolo_cpu_pipeline::YoloCpuPipeline;
use super::yolo_gpu_pipeline::YoloGpuPipeline;
#[cfg(feature = "openvino")]
use super::yolo_openvino_pipeline::YoloOpenVinoPipeline;
use super::yolo_model_config::YoloModelConfig;
use super::yolo_pipeline_config::YoloPipelineConfig;
use super::paddleocr_model_config::PaddleOcrModelConfig;
use super::paddleocr_pipeline_config::PaddleOcrPipelineConfig;
use super::paddleocr_pipeline::PaddleOcrPipeline;
#[cfg(feature = "openvino")]
use super::openvino_model_config::OpenVinoModelConfig;
#[cfg(feature = "openvino")]
use super::openvino_ocr_model_config::OpenVinoOcrModelConfig;
#[cfg(feature = "openvino")]
use super::openvino_pipeline_config::OpenVinoPipelineConfig;
#[cfg(feature = "openvino")]
use super::openvino_tensor_pipeline::OpenVinoTensorPipeline;
#[cfg(feature = "openvino")]
use super::openvino_ocr_pipeline::OpenVinoOcrPipeline;
use super::pipeline_instance::PipelineInstance;

#[no_mangle]
pub extern "C" fn sdk_model_manager_create(out_handle: *mut SdkModelManagerHandle) -> SdkErr {
    with_panic_boundary_err("sdk_model_manager_create", || {
        if out_handle.is_null() {
            set_last_error("out_handle is null");
            return SdkErr::InvalidArg;
        }

        let state = ModelManagerState::default();
        let mut managers = match lock_model_managers() {
            Ok(m) => m,
            Err(err) => return err,
        };
        let handle = managers.insert(state);
        unsafe {
            *out_handle = handle;
        }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_model_manager_destroy(handle: SdkModelManagerHandle) -> SdkErr {
    with_panic_boundary_err("sdk_model_manager_destroy", || {
        let mut managers = match lock_model_managers() {
            Ok(m) => m,
            Err(err) => return err,
        };
        let state = match managers.remove(handle) {
            Some(s) => s,
            None => {
                set_last_error("invalid model manager handle");
                return SdkErr::InvalidArg;
            }
        };

        let mut models = match lock_models() {
            Ok(m) => m,
            Err(err) => return err,
        };
        for (_, model_handle) in state.models_by_id {
            let _ = models.remove(model_handle);
        }

        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_model_manager_load_model(
    manager_handle: SdkModelManagerHandle,
    spec: *const SdkModelSpec,
    out_model: *mut SdkModelHandle,
) -> SdkErr {
    with_panic_boundary_err("sdk_model_manager_load_model", || {
        if spec.is_null() || out_model.is_null() {
            set_last_error("spec or out_model is null");
            return SdkErr::InvalidArg;
        }

        let spec = unsafe { &*spec };
        let id = match read_utf8(spec.id_ptr, spec.id_len, "model id") {
            Ok(value) => value,
            Err(err) => return err,
        };
        if id.is_empty() {
            set_last_error("model id is empty");
            return SdkErr::InvalidArg;
        }

        let config_json = match read_utf8(spec.config_json_ptr, spec.config_json_len, "model config") {
            Ok(value) => value,
            Err(err) => return err,
        };

        let mut managers = match lock_model_managers() {
            Ok(m) => m,
            Err(err) => return err,
        };
        let manager = match managers.get_mut(manager_handle) {
            Some(m) => m,
            None => {
                set_last_error("invalid model manager handle");
                return SdkErr::InvalidArg;
            }
        };

        if let Some(existing) = manager.models_by_id.get(&id) {
            unsafe { *out_model = *existing; }
            clear_last_error();
            return SdkErr::Ok;
        }

    let model_instance = match build_model_instance(spec.kind, &config_json) {
            Ok(instance) => instance,
            Err(err) => return err,
        };

        let mut models = match lock_models() {
            Ok(m) => m,
            Err(err) => return err,
        };
        let model_handle = models.insert(model_instance);
        manager.models_by_id.insert(id, model_handle);

        unsafe { *out_model = model_handle; }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_model_create_pipeline(
    model_handle: SdkModelHandle,
    spec: *const SdkPipelineSpec,
    out_pipeline: *mut SdkPipelineHandle,
) -> SdkErr {
    with_panic_boundary_err("sdk_model_create_pipeline", || {
        if spec.is_null() || out_pipeline.is_null() {
            set_last_error("spec or out_pipeline is null");
            return SdkErr::InvalidArg;
        }

        let spec = unsafe { &*spec };
        let _id = match read_utf8(spec.id_ptr, spec.id_len, "pipeline id") {
            Ok(value) => value,
            Err(err) => return err,
        };

        let config_json = match read_utf8(spec.config_json_ptr, spec.config_json_len, "pipeline config") {
            Ok(value) => value,
            Err(err) => return err,
        };

        let mut models = match lock_models() {
            Ok(m) => m,
            Err(err) => return err,
        };
        let model = match models.get_mut(model_handle) {
            Some(m) => m,
            None => {
                set_last_error("invalid model handle");
                return SdkErr::InvalidArg;
            }
        };

        let pipeline_instance = match build_pipeline_instance(model, spec.kind, &config_json) {
            Ok(instance) => instance,
            Err(err) => return err,
        };

        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        let pipeline_handle = pipelines.insert(pipeline_instance);
        unsafe { *out_pipeline = pipeline_handle; }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_pipeline_destroy(handle: SdkPipelineHandle) -> SdkErr {
    with_panic_boundary_err("sdk_pipeline_destroy", || {
        let mut pipelines = match lock_pipelines() {
            Ok(p) => p,
            Err(err) => return err,
        };
        match pipelines.remove(handle) {
            Some(_) => {
                clear_last_error();
                SdkErr::Ok
            }
            None => {
                set_last_error("invalid pipeline handle");
                SdkErr::InvalidArg
            }
        }
    })
}

fn lock_model_managers(
) -> Result<std::sync::MutexGuard<'static, crate::runtime::HandleManager<ModelManagerState>>, SdkErr> {
    SDK_MODEL_MANAGERS.lock().map_err(|_| {
        set_last_error("model manager lock poisoned");
        SdkErr::Runtime
    })
}

fn lock_models(
) -> Result<std::sync::MutexGuard<'static, crate::runtime::HandleManager<ModelInstance>>, SdkErr> {
    SDK_MODELS.lock().map_err(|_| {
        set_last_error("model lock poisoned");
        SdkErr::Runtime
    })
}

fn lock_pipelines(
) -> Result<std::sync::MutexGuard<'static, crate::runtime::HandleManager<PipelineInstance>>, SdkErr> {
    SDK_PIPELINES.lock().map_err(|_| {
        set_last_error("pipeline lock poisoned");
        SdkErr::Runtime
    })
}

fn build_model_instance(kind: ModelKind, config_json: &str) -> Result<ModelInstance, SdkErr> {
    match kind {
        ModelKind::Yolo => {
            let config: YoloModelConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse Yolo model config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };
            if config.model_path.is_empty() {
                set_last_error("model_path is required");
                return Err(SdkErr::InvalidArg);
            }
            Ok(ModelInstance {
                kind,
                yolo: Some(config),
                paddleocr: None,
                #[cfg(feature = "openvino")]
                openvino: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        ModelKind::PaddleOcr => {
            let config: PaddleOcrModelConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse PaddleOCR model config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };
            if config.det_model_dir.is_empty() || config.rec_model_dir.is_empty() {
                set_last_error("det_model_dir and rec_model_dir are required");
                return Err(SdkErr::InvalidArg);
            }
            Ok(ModelInstance {
                kind,
                yolo: None,
                paddleocr: Some(config),
                #[cfg(feature = "openvino")]
                openvino: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        #[cfg(feature = "openvino")]
        ModelKind::OpenVino => {
            let config: OpenVinoModelConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse OpenVINO model config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };
            if config.model_path.is_empty() {
                set_last_error("model_path is required");
                return Err(SdkErr::InvalidArg);
            }
            Ok(ModelInstance {
                kind,
                yolo: None,
                paddleocr: None,
                openvino: Some(config),
                openvino_ocr: None,
            })
        }
        #[cfg(feature = "openvino")]
        ModelKind::OpenVinoOcr => {
            let config: OpenVinoOcrModelConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse OpenVINO OCR model config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };
            if config.det_model_path.is_empty() || config.rec_model_path.is_empty() || config.dict_path.is_empty() {
                set_last_error("det_model_path, rec_model_path, and dict_path are required");
                return Err(SdkErr::InvalidArg);
            }
            Ok(ModelInstance {
                kind,
                yolo: None,
                paddleocr: None,
                openvino: None,
                openvino_ocr: Some(config),
            })
        }
        #[cfg(not(feature = "openvino"))]
        ModelKind::OpenVino => {
            set_last_error("OpenVINO not built in this configuration");
            Err(SdkErr::Unsupported)
        }
        #[cfg(not(feature = "openvino"))]
        ModelKind::OpenVinoOcr => {
            set_last_error("OpenVINO not built in this configuration");
            Err(SdkErr::Unsupported)
        }
        _ => {
            set_last_error("unsupported model kind");
            Err(SdkErr::Unsupported)
        }
    }
}

fn build_pipeline_instance(model: &mut ModelInstance, kind: PipelineKind, config_json: &str) -> Result<PipelineInstance, SdkErr> {
    match (model.kind, kind) {
        (ModelKind::Yolo, PipelineKind::YoloGpuThroughput) => {
            let config: YoloPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse Yolo pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };
            let yolo_config = match model.yolo.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing Yolo model config");
                    return Err(SdkErr::BadState);
                }
            };

            let pipeline = match YoloGpuPipeline::new(yolo_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: None,
                yolo_gpu: Some(pipeline),
                #[cfg(feature = "openvino")]
                yolo_openvino: None,
                paddleocr: None,
                #[cfg(feature = "openvino")]
                openvino_tensor: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        (ModelKind::Yolo, PipelineKind::YoloCpu) => {
            let yolo_config = match model.yolo.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing Yolo model config");
                    return Err(SdkErr::BadState);
                }
            };

            let config: YoloPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse Yolo pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };

            let pipeline = match YoloCpuPipeline::new(yolo_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: Some(pipeline),
                yolo_gpu: None,
                #[cfg(feature = "openvino")]
                yolo_openvino: None,
                paddleocr: None,
                #[cfg(feature = "openvino")]
                openvino_tensor: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        #[cfg(feature = "openvino")]
        (ModelKind::Yolo, PipelineKind::YoloOpenVino) => {
            let yolo_config = match model.yolo.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing Yolo model config");
                    return Err(SdkErr::BadState);
                }
            };

            let config: YoloPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse Yolo pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };

            let pipeline = match YoloOpenVinoPipeline::new(yolo_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: None,
                yolo_gpu: None,
                #[cfg(feature = "openvino")]
                yolo_openvino: Some(pipeline),
                paddleocr: None,
                #[cfg(feature = "openvino")]
                openvino_tensor: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        #[cfg(not(feature = "openvino"))]
        (ModelKind::Yolo, PipelineKind::YoloOpenVino) => {
            set_last_error("OpenVINO not built in this configuration");
            Err(SdkErr::Unsupported)
        }
        (ModelKind::PaddleOcr, PipelineKind::PaddleOcr) => {
            let config: PaddleOcrPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse PaddleOCR pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };

            let ocr_config = match model.paddleocr.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing PaddleOCR model config");
                    return Err(SdkErr::BadState);
                }
            };

            let pipeline = match PaddleOcrPipeline::new(ocr_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: None,
                yolo_gpu: None,
                #[cfg(feature = "openvino")]
                yolo_openvino: None,
                paddleocr: Some(pipeline),
                #[cfg(feature = "openvino")]
                openvino_tensor: None,
                #[cfg(feature = "openvino")]
                openvino_ocr: None,
            })
        }
        #[cfg(feature = "openvino")]
        (ModelKind::OpenVino, PipelineKind::OpenVinoTensor) => {
            let config: OpenVinoPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse OpenVINO pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };

            let ov_config = match model.openvino.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing OpenVINO model config");
                    return Err(SdkErr::BadState);
                }
            };

            let pipeline = match OpenVinoTensorPipeline::new(ov_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: None,
                yolo_gpu: None,
                #[cfg(feature = "openvino")]
                yolo_openvino: None,
                paddleocr: None,
                openvino_tensor: Some(pipeline),
                openvino_ocr: None,
            })
        }
        #[cfg(feature = "openvino")]
        (ModelKind::OpenVinoOcr, PipelineKind::OpenVinoOcr) => {
            let config: OpenVinoPipelineConfig = match serde_json::from_str(config_json) {
                Ok(value) => value,
                Err(_) => {
                    set_last_error("failed to parse OpenVINO OCR pipeline config JSON");
                    return Err(SdkErr::InvalidArg);
                }
            };

            let ocr_config = match model.openvino_ocr.as_ref() {
                Some(cfg) => cfg,
                None => {
                    set_last_error("missing OpenVINO OCR model config");
                    return Err(SdkErr::BadState);
                }
            };

            let pipeline = match OpenVinoOcrPipeline::new(ocr_config, &config) {
                Ok(p) => p,
                Err(err) => return Err(err),
            };

            Ok(PipelineInstance {
                yolo_cpu: None,
                yolo_gpu: None,
                #[cfg(feature = "openvino")]
                yolo_openvino: None,
                paddleocr: None,
                openvino_tensor: None,
                openvino_ocr: Some(pipeline),
            })
        }
        #[cfg(not(feature = "openvino"))]
        (ModelKind::OpenVino, PipelineKind::OpenVinoTensor) => {
            set_last_error("OpenVINO not built in this configuration");
            Err(SdkErr::Unsupported)
        }
        #[cfg(not(feature = "openvino"))]
        (ModelKind::OpenVinoOcr, PipelineKind::OpenVinoOcr) => {
            set_last_error("OpenVINO not built in this configuration");
            Err(SdkErr::Unsupported)
        }
        _ => {
            set_last_error("unsupported pipeline kind");
            Err(SdkErr::Unsupported)
        }
    }
}
