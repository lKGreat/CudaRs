#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SdkOcrLine {
    pub points: [f32; 8],
    pub score: f32,
    pub cls_label: i32,
    pub cls_score: f32,
    pub text_offset: u32,
    pub text_len: u32,
}
