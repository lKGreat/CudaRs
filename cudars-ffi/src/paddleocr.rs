use libc::{c_char, c_int, c_uchar, c_void};

#[repr(C)]
pub struct PaddleOcrInitOptions {
    pub doc_orientation_model_name: *const c_char,
    pub doc_orientation_model_dir: *const c_char,
    pub doc_unwarping_model_name: *const c_char,
    pub doc_unwarping_model_dir: *const c_char,
    pub text_detection_model_name: *const c_char,
    pub text_detection_model_dir: *const c_char,
    pub textline_orientation_model_name: *const c_char,
    pub textline_orientation_model_dir: *const c_char,
    pub text_recognition_model_name: *const c_char,
    pub text_recognition_model_dir: *const c_char,
    pub lang: *const c_char,
    pub ocr_version: *const c_char,
    pub vis_font_dir: *const c_char,
    pub device: *const c_char,
    pub precision: *const c_char,
    pub text_det_limit_type: *const c_char,
    pub paddlex_config_yaml: *const c_char,

    pub textline_orientation_batch_size: i32,
    pub text_recognition_batch_size: i32,
    pub use_doc_orientation_classify: i32,
    pub use_doc_unwarping: i32,
    pub use_textline_orientation: i32,
    pub text_det_limit_side_len: i32,
    pub text_det_max_side_limit: i32,
    pub enable_mkldnn: i32,
    pub mkldnn_cache_capacity: i32,
    pub cpu_threads: i32,
    pub thread_num: i32,
    pub enable_struct_json: i32,

    pub text_det_thresh: f32,
    pub text_det_box_thresh: f32,
    pub text_det_unclip_ratio: f32,
    pub text_rec_score_thresh: f32,

    pub text_det_input_shape: [i32; 4],
    pub text_rec_input_shape: [i32; 4],
    pub text_det_input_shape_len: i32,
    pub text_rec_input_shape_len: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaddleOcrLine {
    pub points: [f32; 8],
    pub score: f32,
    pub cls_label: i32,
    pub cls_score: f32,
    pub text_offset: u32,
    pub text_len: u32,
}

pub const PADDLEOCR_OPTION_UNSET_I32: i32 = i32::MIN;

extern "C" {
    pub fn paddleocr_create(options: *const PaddleOcrInitOptions, out_handle: *mut *mut c_void) -> c_int;
    pub fn paddleocr_destroy(handle: *mut c_void) -> c_int;
    pub fn paddleocr_run_image(handle: *mut c_void, data: *const c_uchar, len: usize) -> c_int;
    pub fn paddleocr_get_line_count(handle: *mut c_void, out_count: *mut usize) -> c_int;
    pub fn paddleocr_write_lines(handle: *mut c_void, dst: *mut PaddleOcrLine, cap: usize, out_written: *mut usize) -> c_int;
    pub fn paddleocr_get_text_bytes(handle: *mut c_void, out_bytes: *mut usize) -> c_int;
    pub fn paddleocr_write_text(handle: *mut c_void, dst: *mut c_char, cap: usize, out_written: *mut usize) -> c_int;
    pub fn paddleocr_get_struct_json_bytes(handle: *mut c_void, out_bytes: *mut usize) -> c_int;
    pub fn paddleocr_write_struct_json(handle: *mut c_void, dst: *mut c_char, cap: usize, out_written: *mut usize) -> c_int;
    pub fn paddleocr_last_error(out_len: *mut usize) -> *const c_char;
}
