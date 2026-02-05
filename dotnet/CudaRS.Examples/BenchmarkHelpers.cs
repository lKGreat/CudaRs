namespace CudaRS.Examples;

using System.Diagnostics;

/// <summary>
/// 基准测试辅助工具
/// </summary>
static class BenchmarkHelpers
{
    /// <summary>
    /// 执行基准测试
    /// </summary>
    public static BenchmarkResult RunBenchmark<T>(
        string name,
        Func<ReadOnlyMemory<byte>, T> pipelineFunc,
        List<ImageInput> images,
        int iterations,
        int warmup = 0,
        Action<T, int>? onResult = null)
    {
        if (images.Count == 0)
            throw new ArgumentException("No images provided", nameof(images));

        Console.WriteLine();
        Console.WriteLine($"=== {name} Benchmark ===");
        Console.WriteLine($"Images: {images.Count}, Iterations: {iterations}, Warmup: {warmup}");

        // Warmup
        for (var i = 0; i < warmup; i++)
        {
            var warmupImage = images[i % images.Count];
            _ = pipelineFunc(warmupImage.Bytes);
        }

        // Benchmark
        var times = new List<double>(iterations);
        var successCount = 0;
        var failureCount = 0;

        for (var i = 0; i < iterations; i++)
        {
            var image = images[i % images.Count];
            var sw = Stopwatch.StartNew();
            try
            {
                var result = pipelineFunc(image.Bytes);
                sw.Stop();
                var ms = sw.Elapsed.TotalMilliseconds;
                times.Add(ms);
                successCount++;

                onResult?.Invoke(result, i);
                Console.WriteLine($"Iter {i + 1}/{iterations}: {ms:F2} ms - {Path.GetFileName(image.Path)}");
            }
            catch (Exception ex)
            {
                sw.Stop();
                var ms = sw.Elapsed.TotalMilliseconds;
                times.Add(ms);
                failureCount++;
                Console.WriteLine($"Iter {i + 1}/{iterations}: FAILED {ms:F2} ms - {ex.Message}");
            }
        }

        return new BenchmarkResult
        {
            Name = name,
            Iterations = iterations,
            Times = times,
            SuccessCount = successCount,
            FailureCount = failureCount
        };
    }

    /// <summary>
    /// 打印统计信息
    /// </summary>
    public static void PrintStats(BenchmarkResult result)
    {
        Console.WriteLine();
        Console.WriteLine($"=== {result.Name} Statistics ===");
        Console.WriteLine($"Total: {result.TotalMs:F2} ms");
        Console.WriteLine($"Average: {result.Avg:F2} ms");
        Console.WriteLine($"Median: {result.Median:F2} ms");
        Console.WriteLine($"First: {result.FirstMs:F2} ms");
        Console.WriteLine($"Steady Avg: {result.SteadyAvg:F2} ms");
        Console.WriteLine($"Success: {result.SuccessCount}/{result.Iterations}");
        if (result.FailureCount > 0)
            Console.WriteLine($"Failures: {result.FailureCount}");
    }

    /// <summary>
    /// 计算中位数
    /// </summary>
    public static double ComputeMedian(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
            return 0;
        var ordered = values.OrderBy(v => v).ToArray();
        var mid = ordered.Length / 2;
        if (ordered.Length % 2 == 0)
            return (ordered[mid - 1] + ordered[mid]) / 2.0;
        return ordered[mid];
    }
}
