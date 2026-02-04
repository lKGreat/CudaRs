#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineKind {
    Unknown = 0,
    YoloCpu = 1,
    YoloGpuThroughput = 2,
    PaddleOcr = 3,
}
