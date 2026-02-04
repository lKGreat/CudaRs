use super::yolo_cpu_pipeline::YoloCpuPipeline;
use super::yolo_gpu_pipeline::YoloGpuPipeline;
use super::paddleocr_pipeline::PaddleOcrPipeline;

pub struct PipelineInstance {
    pub yolo_cpu: Option<YoloCpuPipeline>,
    pub yolo_gpu: Option<YoloGpuPipeline>,
    pub paddleocr: Option<PaddleOcrPipeline>,
}
