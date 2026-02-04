use std::mem::size_of;

#[derive(Debug, Clone)]
pub struct OpenVinoOutput {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl OpenVinoOutput {
    pub fn bytes(&self) -> usize {
        self.data.len() * size_of::<f32>()
    }
}
