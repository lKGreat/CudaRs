using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case12Int8Test
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 12] INT8 Quantized Model Support");
        Console.WriteLine("=======================================");

        var modelPath = Config.TestYoloModel;
        
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"⚠ Test model not found: {modelPath}");
            Console.WriteLine($"  Please set Config.TestYoloModel to a valid ONNX model path");
            return;
        }

        // Check if INT8 model exists
        var int8ModelPath = modelPath.Replace(".onnx", "_int8.xml");
        var hasInt8Model = File.Exists(int8ModelPath);

        if (!hasInt8Model)
        {
            Console.WriteLine($"Note: INT8 model not found at {int8ModelPath}");
            Console.WriteLine("  Testing INT8 precision hint with FP32 model instead");
            int8ModelPath = modelPath;
        }

        try
        {
            // ===== Load FP32 Model =====
            Console.WriteLine("\n--- Loading FP32 Model ---");
            var fp32Config = new OpenVinoModelConfig
            {
                ModelPath = modelPath,
                Precision = ModelPrecision.FP32,
            };

            using var fp32Model = new OpenVinoModel("fp32_model", fp32Config);
            Console.WriteLine($"✓ Loaded FP32 model: {Path.GetFileName(modelPath)}");

            var fp32Inputs = fp32Model.GetInputs();
            if (fp32Inputs.Length == 0)
            {
                Console.WriteLine("⚠ Model has no inputs");
                return;
            }

            var inputShape = fp32Inputs[0].Shape;
            Console.WriteLine($"  Input shape: [{string.Join(", ", inputShape)}]");
            Console.WriteLine($"  Element type: {fp32Inputs[0].ElementType}");

            // ===== Load INT8 Model =====
            Console.WriteLine($"\n--- Loading INT8 Model ---");
            var int8Config = new OpenVinoModelConfig
            {
                ModelPath = int8ModelPath,
                Precision = ModelPrecision.INT8,
            };

            using var int8Model = new OpenVinoModel("int8_model", int8Config);
            Console.WriteLine($"✓ Loaded INT8 model: {Path.GetFileName(int8ModelPath)}");

            var int8Inputs = int8Model.GetInputs();
            if (int8Inputs.Length > 0)
            {
                Console.WriteLine($"  Input shape: [{string.Join(", ", int8Inputs[0].Shape)}]");
                Console.WriteLine($"  Element type: {int8Inputs[0].ElementType}");
            }

            // ===== Create Test Data =====
            long totalElements = 1;
            foreach (var dim in inputShape)
            {
                totalElements *= dim;
            }

            var testData = new float[totalElements];
            var random = new Random(42);
            for (int i = 0; i < testData.Length; i++)
            {
                testData[i] = (float)random.NextDouble();
            }

            Console.WriteLine($"\nTest data: {totalElements} elements ({totalElements * sizeof(float) / 1024.0:F2} KB)");

            // ===== Benchmark FP32 Model =====
            Console.WriteLine("\n--- Benchmarking FP32 Model ---");
            using (var fp32Pipeline = fp32Model.CreatePipeline("fp32_pipeline"))
            {
                // Warmup
                for (int i = 0; i < 3; i++)
                {
                    fp32Pipeline.Run(testData, inputShape);
                }

                // Measure
                const int iterations = 10;
                var fp32Times = new double[iterations];
                
                for (int i = 0; i < iterations; i++)
                {
                    var sw = Stopwatch.StartNew();
                    fp32Pipeline.Run(testData, inputShape);
                    sw.Stop();
                    fp32Times[i] = sw.Elapsed.TotalMilliseconds;
                }

                var fp32AvgTime = fp32Times.Average();
                var fp32MinTime = fp32Times.Min();
                
                Console.WriteLine($"Average time: {fp32AvgTime:F2} ms");
                Console.WriteLine($"Min time: {fp32MinTime:F2} ms");
                Console.WriteLine($"Throughput: {1000.0 / fp32AvgTime:F2} inferences/sec");
            }

            // ===== Benchmark INT8 Model =====
            Console.WriteLine("\n--- Benchmarking INT8 Model ---");
            using (var int8Pipeline = int8Model.CreatePipeline("int8_pipeline"))
            {
                // Warmup
                for (int i = 0; i < 3; i++)
                {
                    int8Pipeline.Run(testData, inputShape);
                }

                // Measure
                const int iterations = 10;
                var int8Times = new double[iterations];
                
                for (int i = 0; i < iterations; i++)
                {
                    var sw = Stopwatch.StartNew();
                    int8Pipeline.Run(testData, inputShape);
                    sw.Stop();
                    int8Times[i] = sw.Elapsed.TotalMilliseconds;
                }

                var int8AvgTime = int8Times.Average();
                var int8MinTime = int8Times.Min();
                
                Console.WriteLine($"Average time: {int8AvgTime:F2} ms");
                Console.WriteLine($"Min time: {int8MinTime:F2} ms");
                Console.WriteLine($"Throughput: {1000.0 / int8AvgTime:F2} inferences/sec");
            }

            // ===== Performance Comparison =====
            Console.WriteLine("\n--- Performance Comparison ---");
            var fp32Avg = 0.0;
            var int8Avg = 0.0;

            using (var fp32Pipeline = fp32Model.CreatePipeline("fp32_pipeline"))
            using (var int8Pipeline = int8Model.CreatePipeline("int8_pipeline"))
            {
                // Quick measurement
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 5; i++)
                {
                    fp32Pipeline.Run(testData, inputShape);
                }
                sw.Stop();
                fp32Avg = sw.Elapsed.TotalMilliseconds / 5;

                sw.Restart();
                for (int i = 0; i < 5; i++)
                {
                    int8Pipeline.Run(testData, inputShape);
                }
                sw.Stop();
                int8Avg = sw.Elapsed.TotalMilliseconds / 5;
            }

            var speedup = fp32Avg / int8Avg;
            var improvement = (fp32Avg - int8Avg) / fp32Avg * 100;

            Console.WriteLine($"FP32 average: {fp32Avg:F2} ms");
            Console.WriteLine($"INT8 average: {int8Avg:F2} ms");
            Console.WriteLine($"Speedup: {speedup:F2}x");
            Console.WriteLine($"Improvement: {improvement:F1}%");

            if (!hasInt8Model)
            {
                Console.WriteLine("\nNote: Tested with FP32 model using INT8 precision hint");
                Console.WriteLine("  For true INT8 performance, quantize the model using:");
                Console.WriteLine("    - OpenVINO Model Optimizer (mo)");
                Console.WriteLine("    - Neural Network Compression Framework (NNCF)");
                Console.WriteLine("    - Post-training Optimization Tool (POT)");
            }

            Console.WriteLine("\n✓ Case 12 通过: Successfully loaded and benchmarked INT8 model");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Case 12 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
