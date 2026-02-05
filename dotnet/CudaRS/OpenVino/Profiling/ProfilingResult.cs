using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.OpenVino.Profiling;

/// <summary>
/// Performance profiling results from OpenVINO inference.
/// </summary>
public sealed class ProfilingResult
{
    /// <summary>
    /// Gets whether profiling was enabled.
    /// </summary>
    public bool ProfilingEnabled { get; init; }

    /// <summary>
    /// Gets the total inference time.
    /// </summary>
    public TimeSpan TotalTime { get; init; }

    /// <summary>
    /// Gets the number of layers profiled.
    /// </summary>
    public int LayerCount { get; init; }

    /// <summary>
    /// Gets the execution time for each layer (layer name -> time).
    /// </summary>
    public Dictionary<string, TimeSpan> LayerTimes { get; init; } = new();

    /// <summary>
    /// Prints a formatted summary of profiling results.
    /// </summary>
    public void Print()
    {
        Console.WriteLine("=== Profiling Results ===");
        
        if (!ProfilingEnabled)
        {
            Console.WriteLine("Profiling was not enabled");
            return;
        }

        if (TotalTime > TimeSpan.Zero)
        {
            Console.WriteLine($"Total time: {TotalTime.TotalMilliseconds:F2} ms");
        }

        if (LayerCount > 0)
        {
            Console.WriteLine($"Layer count: {LayerCount}");
        }

        if (LayerTimes.Count > 0)
        {
            Console.WriteLine("\nLayer timings:");
            var sortedLayers = LayerTimes.OrderByDescending(kv => kv.Value).ToList();
            
            foreach (var (layer, time) in sortedLayers.Take(10))
            {
                Console.WriteLine($"  {layer}: {time.TotalMilliseconds:F3} ms");
            }

            if (sortedLayers.Count > 10)
            {
                Console.WriteLine($"  ... and {sortedLayers.Count - 10} more layers");
            }
        }
    }

    /// <summary>
    /// Gets the top N slowest layers.
    /// </summary>
    public IEnumerable<(string Layer, TimeSpan Time)> GetSlowestLayers(int count = 10)
    {
        return LayerTimes
            .OrderByDescending(kv => kv.Value)
            .Take(count)
            .Select(kv => (kv.Key, kv.Value));
    }
}
