#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdkErr {
    Ok = 0,
    InvalidArg = 1,
    OutOfMemory = 2,
    Runtime = 3,
    Unsupported = 4,
    NotFound = 5,
    Timeout = 6,
    Busy = 7,
    Io = 8,
    Permission = 9,
    Canceled = 10,
    BadState = 11,
    VersionMismatch = 12,
    Backend = 13,
    MissingDependency = 14,
    DllNotFound = 15,
    ModelLoadFailed = 16,
    ConfigInvalid = 17,
}

impl Default for SdkErr {
    fn default() -> Self {
        SdkErr::Ok
    }
}
