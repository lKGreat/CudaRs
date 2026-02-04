use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OpenVinoPipelineConfig {
    pub openvino_device: String,
    pub openvino_config_json: String,
}

impl Default for OpenVinoPipelineConfig {
    fn default() -> Self {
        Self {
            openvino_device: "auto".to_string(),
            openvino_config_json: String::new(),
        }
    }
}
