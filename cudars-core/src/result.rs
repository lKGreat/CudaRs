use crate::SdkErr;

pub type SdkResult<T> = core::result::Result<T, SdkErr>;
