use crate::PipelineKind;

pub trait Pipeline: Send {
    fn kind(&self) -> PipelineKind;
}
