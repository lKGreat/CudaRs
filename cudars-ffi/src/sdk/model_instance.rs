use cudars_core::ModelKind;

use super::yolo_model_config::YoloModelConfig;

pub struct ModelInstance {
    pub id: String,
    pub kind: ModelKind,
    pub yolo: Option<YoloModelConfig>,
}
