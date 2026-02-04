use cudars_core::ModelKind;

use super::yolo_model_config::YoloModelConfig;
use super::paddleocr_model_config::PaddleOcrModelConfig;
use super::openvino_model_config::OpenVinoModelConfig;

pub struct ModelInstance {
    pub kind: ModelKind,
    pub yolo: Option<YoloModelConfig>,
    pub paddleocr: Option<PaddleOcrModelConfig>,
    pub openvino: Option<OpenVinoModelConfig>,
}
