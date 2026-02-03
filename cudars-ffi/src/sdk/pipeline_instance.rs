use cudars_core::PipelineKind;

use super::yolo_cpu_pipeline::YoloCpuPipeline;
use super::yolo_gpu_pipeline::YoloGpuPipeline;

pub struct PipelineInstance {
    pub kind: PipelineKind,
    pub yolo_cpu: Option<YoloCpuPipeline>,
    pub yolo_gpu: Option<YoloGpuPipeline>,
}
