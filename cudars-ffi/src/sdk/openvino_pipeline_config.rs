use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OpenVinoPipelineConfig {
    pub openvino_device: String,
    pub openvino_config_json: String,
    pub openvino_performance_mode: String,
    pub openvino_num_requests: i32,
    pub openvino_num_streams: i32,
    pub openvino_enable_profiling: bool,
    pub openvino_cache_dir: String,
    pub openvino_enable_mmap: Option<bool>,
}

impl Default for OpenVinoPipelineConfig {
    fn default() -> Self {
        Self {
            openvino_device: "auto".to_string(),
            openvino_config_json: String::new(),
            openvino_performance_mode: String::new(),
            openvino_num_requests: 0,
            openvino_num_streams: 0,
            openvino_enable_profiling: false,
            openvino_cache_dir: String::new(),
            openvino_enable_mmap: None,
        }
    }
}
