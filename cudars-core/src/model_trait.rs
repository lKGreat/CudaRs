use crate::{ModelKind, Pipeline, PipelineSpec, SdkResult};

pub trait Model: Send {
    fn id(&self) -> &str;
    fn kind(&self) -> ModelKind;
    fn create_pipeline(&self, spec: PipelineSpec) -> SdkResult<Box<dyn Pipeline>>;
}
