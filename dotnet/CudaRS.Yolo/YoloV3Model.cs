using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV3Model : YoloModelBase
{
    public YoloV3Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V3), hub)
    {
    }
}
