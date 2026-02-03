use std::sync::Mutex;

use crate::runtime::HandleManager;

use super::model_manager_state::ModelManagerState;
use super::model_instance::ModelInstance;
use super::pipeline_instance::PipelineInstance;

pub type SdkModelManagerHandle = u64;
pub type SdkModelHandle = u64;
pub type SdkPipelineHandle = u64;

lazy_static::lazy_static! {
    pub static ref SDK_MODEL_MANAGERS: Mutex<HandleManager<ModelManagerState>> = Mutex::new(HandleManager::new());
    pub static ref SDK_MODELS: Mutex<HandleManager<ModelInstance>> = Mutex::new(HandleManager::new());
    pub static ref SDK_PIPELINES: Mutex<HandleManager<PipelineInstance>> = Mutex::new(HandleManager::new());
}
