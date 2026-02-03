using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV5Model : YoloModelBase
{
    public YoloV5Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V5), hub)
    {
    }
}
