using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV6Model : YoloModelBase
{
    public YoloV6Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V6), hub)
    {
    }
}
