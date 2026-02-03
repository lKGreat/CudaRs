use crate::{Model, ModelSpec, SdkResult};

pub trait ModelManager: Send + Sync {
    fn load_model(&self, spec: ModelSpec) -> SdkResult<Box<dyn Model>>;
}
