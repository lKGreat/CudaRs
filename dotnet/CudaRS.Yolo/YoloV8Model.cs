using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV8Model : YoloModelBase
{
    public YoloV8Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V8), hub)
    {
    }
}
