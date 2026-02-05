using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case5BasicBatchTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 5] Basic Batch Inference Test");
        Console.WriteLine("=====================================");

        var modelPath = Config.TestYoloModel;
        
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"⚠ Test model not found: {modelPath}");
            Console.WriteLine($"  Please set Config.TestYoloModel to a valid ONNX model path");
            return;
        }

        try
        {
            var config = new OpenVinoModelConfig
            {
                ModelPath = modelPath,
            };

            using var model = new OpenVinoModel("yolo11n_batch", config);
            using var pipeline = model.CreatePipeline("cpu");

        // Test parameters
        const int batchSize = 4;
        const int channels = 3;
        const int height = 640;
        const int width = 640;
        const int singleSize = channels * height * width;

        Console.WriteLine($"\nCreating {batchSize} random tensors of shape [{channels}, {height}, {width}]");

        // Create random input tensors
        var random = new Random(42);
        var inputs = new ReadOnlyMemory<float>[batchSize];
        
        for (int i = 0; i < batchSize; i++)
        {
            var data = new float[singleSize];
            for (int j = 0; j < singleSize; j++)
            {
                data[j] = (float)random.NextDouble();
            }
            inputs[i] = data;
        }

        var singleShape = new long[] { channels, height, width };

        // Warmup
        Console.WriteLine("\nWarming up...");
        _ = pipeline.RunBatch(inputs, singleShape);

            // Test batch inference
            Console.WriteLine("\nRunning batch inference...");
            var sw = Stopwatch.StartNew();
            var results = pipeline.RunBatch(inputs, singleShape);
            sw.Stop();
            var batchTime = sw.ElapsedMilliseconds;

            // Validate results
            if (results.Length != batchSize)
            {
                throw new Exception($"Expected {batchSize} results, got {results.Length}");
            }

            Console.WriteLine($"\nBatch inference completed in {batchTime}ms");
            Console.WriteLine($"Throughput: {batchSize * 1000.0 / batchTime:F2} imgs/sec");

            for (int i = 0; i < batchSize; i++)
            {
                Console.WriteLine($"\nBatch item {i}: {results[i].Length} outputs");
                for (int j = 0; j < results[i].Length; j++)
                {
                    var output = results[i][j];
                    var shapeStr = string.Join(", ", output.Shape);
                    Console.WriteLine($"  Output {j}: shape=[{shapeStr}], size={output.Data.Length}");
                }
            }

            // Compare with single inference throughput
            Console.WriteLine("\n\nComparing with single inference...");
            var singleTimes = new long[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                sw.Restart();
                _ = pipeline.Run(inputs[i], new long[] { 1, channels, height, width });
                sw.Stop();
                singleTimes[i] = sw.ElapsedMilliseconds;
            }

            var avgSingleTime = singleTimes.Average();
            var totalSingleTime = singleTimes.Sum();
            
            Console.WriteLine($"Single inference (x{batchSize}): {totalSingleTime}ms total ({avgSingleTime:F2}ms avg)");
            Console.WriteLine($"Batch inference: {batchTime}ms");
            
            // Note: For CPU inference, batch may not always be faster than sequential single inferences
            // The benefit is more significant on GPU or with certain model architectures
            Console.WriteLine($"\nSpeedup: {totalSingleTime / (double)batchTime:F2}x");
            
            Console.WriteLine("\n✓ Case 5 通过: Batch inference working correctly");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Case 5 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
