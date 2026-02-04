use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OpenVinoModelConfig {
    pub model_path: String,
}

impl Default for OpenVinoModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
        }
    }
}
