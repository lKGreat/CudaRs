using System.Text.Json;

namespace CudaRS.Yolo;

internal static class YoloModelConfigJson
{
    public static string Build(YoloModelDefinition definition)
    {
        var backend = definition.Config.Backend.ToString().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(backend) || backend == "auto")
            backend = "tensorrt";

        var dto = new YoloModelConfigDto
        {
            ModelPath = definition.ModelPath,
            DeviceId = definition.DeviceId,
            InputWidth = definition.Config.InputWidth,
            InputHeight = definition.Config.InputHeight,
            InputChannels = definition.Config.InputChannels,
            Backend = backend,
        };

        return JsonSerializer.Serialize(dto);
    }
}
