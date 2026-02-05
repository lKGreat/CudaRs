namespace CudaRS.Examples;

using CudaRS.Yolo;

/// <summary>
/// 文件路径处理和环境设置辅助工具
/// </summary>
static class PathHelpers
{
    /// <summary>
    /// 查找模型文件
    /// </summary>
    public static List<string> FindModels(IEnumerable<string> candidates, string extension)
    {
        var results = new List<string>();
        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
                continue;

            var expanded = Environment.ExpandEnvironmentVariables(candidate.Trim());
            if (Directory.Exists(expanded))
            {
                results.AddRange(Directory.EnumerateFiles(expanded, $"*{extension}", SearchOption.TopDirectoryOnly));
                continue;
            }

            if (File.Exists(expanded))
            {
                results.Add(expanded);
                continue;
            }

            if (HasWildcard(expanded))
            {
                var dir = Path.GetDirectoryName(expanded);
                if (string.IsNullOrWhiteSpace(dir))
                    dir = Directory.GetCurrentDirectory();
                var pattern = Path.GetFileName(expanded);
                if (!string.IsNullOrWhiteSpace(pattern) && Directory.Exists(dir))
                    results.AddRange(Directory.EnumerateFiles(dir, pattern, SearchOption.TopDirectoryOnly));
            }
        }

        return results
            .Where(File.Exists)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>
    /// 加载图片文件
    /// </summary>
    public static List<ImageInput> LoadImages(IEnumerable<string> candidates)
    {
        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png"
        };

        var paths = new List<string>();
        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
                continue;

            var expanded = Environment.ExpandEnvironmentVariables(candidate.Trim());
            if (Directory.Exists(expanded))
            {
                paths.AddRange(Directory.EnumerateFiles(expanded)
                    .Where(p => exts.Contains(Path.GetExtension(p))));
                continue;
            }

            if (File.Exists(expanded))
            {
                if (exts.Contains(Path.GetExtension(expanded)))
                    paths.Add(expanded);
                continue;
            }

            if (HasWildcard(expanded))
            {
                var dir = Path.GetDirectoryName(expanded);
                if (string.IsNullOrWhiteSpace(dir))
                    dir = Directory.GetCurrentDirectory();
                var pattern = Path.GetFileName(expanded);
                if (!string.IsNullOrWhiteSpace(pattern) && Directory.Exists(dir))
                    paths.AddRange(Directory.EnumerateFiles(dir, pattern, SearchOption.TopDirectoryOnly)
                        .Where(p => exts.Contains(Path.GetExtension(p))));
            }
        }

        var images = new List<ImageInput>();
        foreach (var path in paths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                if (!File.Exists(path))
                    continue;
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length > 0)
                    images.Add(new ImageInput(path, bytes));
            }
            catch
            {
                // Skip unreadable files
            }
        }

        return images;
    }

    /// <summary>
    /// 加载标签文件
    /// </summary>
    public static string[] LoadLabels(string modelPath, string? overridePath = null)
    {
        if (!string.IsNullOrWhiteSpace(overridePath) && File.Exists(overridePath))
            return YoloLabels.LoadFromFile(overridePath);

        var dir = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrWhiteSpace(dir))
            return Array.Empty<string>();

        var labelsPath = Path.Combine(dir, "labels.txt");
        if (!File.Exists(labelsPath))
            return Array.Empty<string>();

        return YoloLabels.LoadFromFile(labelsPath);
    }

    /// <summary>
    /// 从路径推断 YOLO 版本
    /// </summary>
    public static YoloVersion? InferVersion(string path)
    {
        var name = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        var map = new Dictionary<string, YoloVersion>
        {
            { "v11", YoloVersion.V11 },
            { "v10", YoloVersion.V10 },
            { "v9", YoloVersion.V9 },
            { "v8", YoloVersion.V8 },
            { "v7", YoloVersion.V7 },
            { "v6", YoloVersion.V6 },
            { "v5", YoloVersion.V5 },
            { "v4", YoloVersion.V4 },
            { "v3", YoloVersion.V3 },
        };

        foreach (var pair in map)
        {
            if (name.Contains(pair.Key))
                return pair.Value;
        }

        return null;
    }

    /// <summary>
    /// 确保 CUDA/TensorRT bin 目录在 PATH 中
    /// </summary>
    public static void EnsureCudaBinsOnPath()
    {
        var candidates = new List<string>();

        if (!string.IsNullOrWhiteSpace(Config.CudaRoot))
            candidates.Add(Path.Combine(Config.CudaRoot, "bin"));
        if (!string.IsNullOrWhiteSpace(Config.TensorRtRoot))
        {
            candidates.Add(Path.Combine(Config.TensorRtRoot, "bin"));
            candidates.Add(Path.Combine(Config.TensorRtRoot, "lib"));
        }

        var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        var parts = new HashSet<string>(
            path.Split(';', StringSplitOptions.RemoveEmptyEntries),
            StringComparer.OrdinalIgnoreCase);

        var updated = false;
        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
                continue;
            if (!Directory.Exists(candidate))
                continue;
            if (parts.Add(candidate))
                updated = true;
        }

        if (updated)
        {
            var newPath = string.Join(";", parts);
            Environment.SetEnvironmentVariable("PATH", newPath);
            Console.WriteLine("Updated PATH with CUDA/TensorRT bin directories.");
        }
    }

    private static bool HasWildcard(string path)
    {
        return path.Contains('*') || path.Contains('?');
    }
}
