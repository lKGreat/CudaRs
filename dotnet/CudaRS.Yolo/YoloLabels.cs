using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CudaRS.Yolo;

public static class YoloLabels
{
    public static string[] LoadFromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Labels path is required.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException("Labels file not found.", path);

        var labels = new List<string>();
        foreach (var line in File.ReadLines(path))
        {
            var label = line.Trim();
            if (string.IsNullOrWhiteSpace(label))
                continue;
            if (label.StartsWith("#", StringComparison.Ordinal))
                continue;
            labels.Add(label);
        }

        return labels.ToArray();
    }
}
