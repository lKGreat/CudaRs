namespace CudaRS;

public enum PipelineKind
{
    Unknown = 0,
    YoloCpu = 1,
    YoloGpuThroughput = 2,
    PaddleOcr = 3,
    YoloOpenVino = 4,
    OpenVinoTensor = 5,
    OpenVinoOcr = 6,
}
