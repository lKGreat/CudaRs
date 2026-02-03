use cudars_core::{ModelKind, PipelineKind, SdkErr};

use super::model_instance::ModelInstance;
use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err};
use super::sdk_handles::{SdkModelHandle, SdkModelManagerHandle, SdkPipelineHandle, SDK_MODEL_MANAGERS, SDK_MODELS, SDK_PIPELINES};
use super::sdk_model_spec::SdkModelSpec;
use super::sdk_pipeline_spec::SdkPipelineSpec;
use super::sdk_strings::read_utf8;
use super::yolo_cpu_pipeline::YoloCpuPipeline;
use super::yolo_gpu_pipeline::YoloGpuPipeline;
use super::yolo_model_config::YoloModelConfig;
use super::yolo_pipeline_config::YoloPipelineConfig;
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

        let model_instance = match build_model_instance(spec.kind, &id, &config_json) {
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

fn build_model_instance(kind: ModelKind, id: &str, config_json: &str) -> Result<ModelInstance, SdkErr> {
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
                id: id.to_string(),
                kind,
                yolo: Some(config),
            })
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
                kind,
                yolo_cpu: None,
                yolo_gpu: Some(pipeline),
            })
        }
        (ModelKind::Yolo, PipelineKind::YoloCpu) => {
            Ok(PipelineInstance {
                kind,
                yolo_cpu: Some(YoloCpuPipeline::new()),
                yolo_gpu: None,
            })
        }
        _ => {
            set_last_error("unsupported pipeline kind");
            Err(SdkErr::Unsupported)
        }
    }
}
