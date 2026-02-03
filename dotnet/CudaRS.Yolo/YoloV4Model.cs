using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV4Model : YoloModelBase
{
    public YoloV4Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V4), hub)
    {
    }
}
