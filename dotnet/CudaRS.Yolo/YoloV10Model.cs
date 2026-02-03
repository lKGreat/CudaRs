using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV10Model : YoloModelBase
{
    public YoloV10Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V10), hub)
    {
    }
}
