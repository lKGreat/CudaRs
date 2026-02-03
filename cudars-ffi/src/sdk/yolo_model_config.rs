use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YoloModelConfig {
    pub model_path: String,
    pub device_id: i32,
    pub input_width: i32,
    pub input_height: i32,
    pub input_channels: i32,
    pub backend: String,
}

impl Default for YoloModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            device_id: 0,
            input_width: 640,
            input_height: 640,
            input_channels: 3,
            backend: "tensorrt".to_string(),
        }
    }
}
