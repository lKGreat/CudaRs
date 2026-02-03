use crate::PipelineKind;

#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub id: String,
    pub kind: PipelineKind,
    pub config_json: String,
}
