use crate::ModelKind;

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String,
    pub kind: ModelKind,
    pub config_json: String,
}
