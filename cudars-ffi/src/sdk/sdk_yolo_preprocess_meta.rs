#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SdkYoloPreprocessMeta {
    pub scale: f32,
    pub pad_x: i32,
    pub pad_y: i32,
    pub original_width: i32,
    pub original_height: i32,
}
