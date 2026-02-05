using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case10BasicProfilingTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 10] Basic Performance Statistics");
        Console.WriteLine("=======================================");

        var modelPath = Config.TestYoloModel;
        
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"⚠ Test model not found: {modelPath}");
            Console.WriteLine($"  Please set Config.TestYoloModel to a valid ONNX model path");
            return;
        }

        try
        {
            // Load model with profiling enabled
            var config = new OpenVinoModelConfig
            {
                ModelPath = modelPath,
                EnableProfiling = true,
            };

            using var model = new OpenVinoModel("test_profiling", config);
            Console.WriteLine($"✓ Loaded model: {Path.GetFileName(modelPath)}");
            Console.WriteLine("  Profiling enabled: Yes");

            // Get input shape
            var inputs = model.GetInputs();
            if (inputs.Length == 0)
            {
                Console.WriteLine("⚠ Model has no inputs");
                return;
            }

            var inputShape = inputs[0].Shape;
            long totalElements = 1;
            foreach (var dim in inputShape)
            {
                totalElements *= dim;
            }

            // Create test data
            var testData = new float[totalElements];
            var random = new Random(42);
            for (int i = 0; i < testData.Length; i++)
            {
                testData[i] = (float)random.NextDouble();
            }

            Console.WriteLine($"\nInput shape: [{string.Join(", ", inputShape)}]");

            // Create pipeline and run inference
            using var pipeline = model.CreatePipeline("test_pipeline");
            
            Console.WriteLine("\nRunning inference...");
            var sw = Stopwatch.StartNew();
            var result = pipeline.Run(testData, inputShape);
            sw.Stop();

            Console.WriteLine($"✓ Inference completed");
            Console.WriteLine($"  Wall-clock time: {sw.Elapsed.TotalMilliseconds:F2} ms");
            Console.WriteLine($"  Outputs: {result.Length}");

            // Get profiling statistics
            Console.WriteLine("\nGetting profiling statistics...");
            var profilingResult = model.GetProfilingInfo();
            
            if (!profilingResult.ProfilingEnabled)
            {
                Console.WriteLine("⚠ Profiling was not enabled (this shouldn't happen)");
                return;
            }

            Console.WriteLine($"✓ Profiling data retrieved");
            Console.WriteLine($"  Profiling enabled: {profilingResult.ProfilingEnabled}");
            Console.WriteLine($"  Layer count: {profilingResult.LayerCount}");

            if (profilingResult.TotalTime > TimeSpan.Zero)
            {
                Console.WriteLine($"  Total time: {profilingResult.TotalTime.TotalMilliseconds:F2} ms");
            }

            // Run multiple iterations for statistical analysis
            Console.WriteLine("\nRunning 10 iterations for performance analysis...");
            var times = new List<double>();
            
            for (int i = 0; i < 10; i++)
            {
                sw.Restart();
                pipeline.Run(testData, inputShape);
                sw.Stop();
                times.Add(sw.Elapsed.TotalMilliseconds);
            }

            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();

            Console.WriteLine($"\nPerformance summary:");
            Console.WriteLine($"  Average: {avgTime:F2} ms");
            Console.WriteLine($"  Min: {minTime:F2} ms");
            Console.WriteLine($"  Max: {maxTime:F2} ms");
            Console.WriteLine($"  Throughput: {1000.0 / avgTime:F2} inferences/sec");

            Console.WriteLine("\n✓ Case 10 通过: Successfully retrieved basic performance statistics");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Case 10 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
