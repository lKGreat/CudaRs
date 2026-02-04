use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YoloPipelineConfig {
    pub max_input_width: i32,
    pub max_input_height: i32,
    pub batch_size: i32,
    pub max_batch_delay_ms: i32,
    pub allow_partial_batch: bool,
    pub worker_count: i32,
    pub cpu_threads: i32,
    pub openvino_device: String,
    pub openvino_config_json: String,
    pub openvino_performance_mode: String,
    pub openvino_num_requests: i32,
    pub openvino_num_streams: i32,
    pub openvino_enable_profiling: bool,
    pub openvino_cache_dir: String,
    pub openvino_enable_mmap: Option<bool>,
}

impl Default for YoloPipelineConfig {
    fn default() -> Self {
        Self {
            max_input_width: 1920,
            max_input_height: 1080,
            batch_size: 1,
            max_batch_delay_ms: 2,
            allow_partial_batch: true,
            worker_count: 1,
            cpu_threads: 1,
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
