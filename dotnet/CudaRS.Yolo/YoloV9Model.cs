using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV9Model : YoloModelBase
{
    public YoloV9Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V9), hub)
    {
    }
}
