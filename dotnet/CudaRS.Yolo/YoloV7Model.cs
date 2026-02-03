using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloV7Model : YoloModelBase
{
    public YoloV7Model(string modelId, string modelPath, YoloConfig? config = null, int deviceId = 0, ModelHub? hub = null)
        : base(BuildDefinition(modelId, modelPath, config, deviceId, YoloVersion.V7), hub)
    {
    }
}
