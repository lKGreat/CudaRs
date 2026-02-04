namespace CudaRS.Native;

public enum SdkPipelineKind : int
{
    Unknown = 0,
    YoloCpu = 1,
    YoloGpuThroughput = 2,
    PaddleOcr = 3,
    YoloOpenVino = 4,
    OpenVinoTensor = 5,
}
