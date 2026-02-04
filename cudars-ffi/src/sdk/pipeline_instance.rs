use super::yolo_cpu_pipeline::YoloCpuPipeline;
use super::yolo_gpu_pipeline::YoloGpuPipeline;
use super::yolo_openvino_pipeline::YoloOpenVinoPipeline;
use super::paddleocr_pipeline::PaddleOcrPipeline;
use super::openvino_tensor_pipeline::OpenVinoTensorPipeline;

pub struct PipelineInstance {
    pub yolo_cpu: Option<YoloCpuPipeline>,
    pub yolo_gpu: Option<YoloGpuPipeline>,
    pub yolo_openvino: Option<YoloOpenVinoPipeline>,
    pub paddleocr: Option<PaddleOcrPipeline>,
    pub openvino_tensor: Option<OpenVinoTensorPipeline>,
}
