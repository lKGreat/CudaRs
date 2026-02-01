using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace CudaRS.Yolo;

public static class YoloModelRegistry
{
    private static readonly ConcurrentDictionary<string, YoloModelDefinition> Definitions = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(YoloModelDefinition definition)
    {
        if (definition == null)
            throw new ArgumentNullException(nameof(definition));
        if (string.IsNullOrWhiteSpace(definition.ModelId))
            throw new ArgumentException("ModelId is required.", nameof(definition));
        if (string.IsNullOrWhiteSpace(definition.ModelPath))
            throw new ArgumentException("ModelPath is required.", nameof(definition));

        Definitions[definition.ModelId] = definition;
    }

    public static bool TryGet(string modelId, out YoloModelDefinition definition)
        => Definitions.TryGetValue(modelId, out definition!);

    public static IReadOnlyDictionary<string, YoloModelDefinition> GetAll()
        => new Dictionary<string, YoloModelDefinition>(Definitions, StringComparer.OrdinalIgnoreCase);
}
