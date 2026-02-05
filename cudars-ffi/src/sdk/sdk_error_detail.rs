use std::cell::RefCell;
use libc::c_char;
use cudars_core::SdkErr;

thread_local! {
    static ERROR_DETAIL: RefCell<ErrorDetail> = RefCell::new(ErrorDetail::default());
}

#[derive(Debug, Clone, Default)]
pub struct ErrorDetail {
    pub code: SdkErr,
    pub message: Vec<u8>,
    pub missing_file: Vec<u8>,
    pub search_paths: Vec<u8>,
    pub suggestion: Vec<u8>,
}

impl ErrorDetail {
    pub fn new(code: SdkErr, message: &str) -> Self {
        Self {
            code,
            message: message.as_bytes().to_vec(),
            missing_file: Vec::new(),
            search_paths: Vec::new(),
            suggestion: Vec::new(),
        }
    }

    pub fn with_missing_file(mut self, file: &str) -> Self {
        self.missing_file = file.as_bytes().to_vec();
        self
    }

    pub fn with_search_paths(mut self, paths: &[String]) -> Self {
        let json = serde_json::to_string(paths).unwrap_or_else(|_| "[]".to_string());
        self.search_paths = json.as_bytes().to_vec();
        self
    }

    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestion = suggestion.as_bytes().to_vec();
        self
    }
}

pub fn set_error_detail(detail: ErrorDetail) {
    ERROR_DETAIL.with(|d| {
        *d.borrow_mut() = detail;
    });
}

pub fn clear_error_detail() {
    ERROR_DETAIL.with(|d| {
        *d.borrow_mut() = ErrorDetail::default();
    });
}

#[repr(C)]
pub struct SdkErrorDetail {
    pub code: SdkErr,
    pub message_ptr: *const c_char,
    pub message_len: usize,
    pub missing_file_ptr: *const c_char,
    pub missing_file_len: usize,
    pub search_paths_ptr: *const c_char,
    pub search_paths_len: usize,
    pub suggestion_ptr: *const c_char,
    pub suggestion_len: usize,
}

#[no_mangle]
pub extern "C" fn sdk_get_error_detail(out: *mut SdkErrorDetail) -> SdkErr {
    if out.is_null() {
        return SdkErr::InvalidArg;
    }

    ERROR_DETAIL.with(|d| {
        let detail = d.borrow();
        unsafe {
            (*out).code = detail.code;
            (*out).message_ptr = if detail.message.is_empty() {
                std::ptr::null()
            } else {
                detail.message.as_ptr() as *const c_char
            };
            (*out).message_len = detail.message.len();
            (*out).missing_file_ptr = if detail.missing_file.is_empty() {
                std::ptr::null()
            } else {
                detail.missing_file.as_ptr() as *const c_char
            };
            (*out).missing_file_len = detail.missing_file.len();
            (*out).search_paths_ptr = if detail.search_paths.is_empty() {
                std::ptr::null()
            } else {
                detail.search_paths.as_ptr() as *const c_char
            };
            (*out).search_paths_len = detail.search_paths.len();
            (*out).suggestion_ptr = if detail.suggestion.is_empty() {
                std::ptr::null()
            } else {
                detail.suggestion.as_ptr() as *const c_char
            };
            (*out).suggestion_len = detail.suggestion.len();
        }
    });

    SdkErr::Ok
}
