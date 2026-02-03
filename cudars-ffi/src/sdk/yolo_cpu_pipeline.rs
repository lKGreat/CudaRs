use cudars_core::SdkErr;

use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;

pub struct YoloCpuPipeline;

impl YoloCpuPipeline {
    pub fn new() -> Self {
        Self
    }

    pub fn run_image(&mut self, _data: *const u8, _len: usize, _meta: *mut SdkYoloPreprocessMeta) -> SdkErr {
        SdkErr::Unsupported
    }

    pub fn output_count(&self) -> usize {
        0
    }
}
