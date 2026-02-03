use libc::c_void;

pub struct YoloOutputBuffer {
    pub device_ptr: *mut c_void,
    pub host_pinned: *mut c_void,
    pub bytes: u64,
    pub shape: Vec<i64>,
}
