use crate::PipelineKind;

#[derive(Debug, Clone)]
pub struct PipelineInfo {
    pub id: String,
    pub kind: PipelineKind,
}
