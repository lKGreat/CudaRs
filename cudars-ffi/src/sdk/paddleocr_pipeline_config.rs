use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PaddleOcrPipelineConfig {
    pub worker_count: i32,
    pub enable_struct_json: bool,
}

impl Default for PaddleOcrPipelineConfig {
    fn default() -> Self {
        Self {
            worker_count: 1,
            enable_struct_json: false,
        }
    }
}
