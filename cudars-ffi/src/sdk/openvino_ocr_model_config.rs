use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OpenVinoOcrModelConfig {
    pub det_model_path: String,
    pub rec_model_path: String,
    pub dict_path: String,
    pub device: Option<String>,
    pub det_resize_long: i32,
    pub det_stride: i32,
    pub det_thresh: f32,
    pub det_box_thresh: f32,
    pub det_unclip_ratio: f32,
    pub det_max_candidates: i32,
    pub det_min_area: i32,
    pub det_box_padding: i32,
    pub rec_input_h: i32,
    pub rec_input_w: i32,
    pub rec_batch_size: i32,
    pub rec_score_thresh: f32,
}

impl Default for OpenVinoOcrModelConfig {
    fn default() -> Self {
        Self {
            det_model_path: String::new(),
            rec_model_path: String::new(),
            dict_path: String::new(),
            device: Some("cpu".to_string()),
            det_resize_long: 960,
            det_stride: 128,
            det_thresh: 0.3,
            det_box_thresh: 0.6,
            det_unclip_ratio: 1.5,
            det_max_candidates: 1000,
            det_min_area: 10,
            det_box_padding: 2,
            rec_input_h: 48,
            rec_input_w: 320,
            rec_batch_size: 8,
            rec_score_thresh: 0.0,
        }
    }
}
