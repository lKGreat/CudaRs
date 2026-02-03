use std::collections::HashMap;

#[derive(Default)]
pub struct ModelManagerState {
    pub models_by_id: HashMap<String, u64>,
}
