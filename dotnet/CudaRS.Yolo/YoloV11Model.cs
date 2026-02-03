using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV11Model : YoloModelBase
{
    public YoloV11Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V11), hub)
    {
    }
}
