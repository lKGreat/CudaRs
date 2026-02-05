use cudars_core::ModelKind;

use super::yolo_model_config::YoloModelConfig;
use super::paddleocr_model_config::PaddleOcrModelConfig;
#[cfg(feature = "openvino")]
use super::openvino_model_config::OpenVinoModelConfig;
#[cfg(feature = "openvino")]
use super::openvino_ocr_model_config::OpenVinoOcrModelConfig;

pub struct ModelInstance {
    pub kind: ModelKind,
    pub yolo: Option<YoloModelConfig>,
    pub paddleocr: Option<PaddleOcrModelConfig>,
    #[cfg(feature = "openvino")]
    pub openvino: Option<OpenVinoModelConfig>,
    #[cfg(feature = "openvino")]
    pub openvino_ocr: Option<OpenVinoOcrModelConfig>,
}
