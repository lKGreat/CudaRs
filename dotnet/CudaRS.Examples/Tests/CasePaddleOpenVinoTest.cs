using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CudaRS.OpenVino;
using CudaRS.Paddle;

namespace CudaRS.Examples.Tests;

/// <summary>
/// Test case for loading PaddlePaddle models with OpenVINO
/// Demonstrates the complete workflow: convert -> load -> infer
/// </summary>
public static class CasePaddleOpenVinoTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case Paddle-OpenVINO] PaddlePaddle Model with OpenVINO");
        Console.WriteLine("=========================================================");

        // Configuration - adjust these paths to your model location
        // Try multiple possible locations
        var possibleDirs = new[]
        {
            @"E:\codeding\AI\PP-OCRv5_mobile_det_infer",
            @"E:\models\PP-OCRv5_mobile_det_infer",
            @"E:\codeding\AI\PaddleOCR\PP-OCRv5_mobile_det_infer",
            Path.Combine(Directory.GetCurrentDirectory(), "models", "PP-OCRv5_mobile_det_infer")
        };

        string? paddleModelDir = null;
        foreach (var dir in possibleDirs)
        {
            if (Directory.Exists(dir))
            {
                paddleModelDir = dir;
                break;
            }
        }

        var modelJsonFile = "inference.json";
        var modelParamsFile = "inference.pdiparams";
        var configYamlFile = "inference.yml";

        // Check if model directory exists
        if (paddleModelDir == null)
        {
            Console.WriteLine($"⚠ PaddlePaddle model directory not found in any of these locations:");
            foreach (var dir in possibleDirs)
            {
                Console.WriteLine($"  - {dir}");
            }
            Console.WriteLine("\nTo run this test:");
            Console.WriteLine("1. Download a PaddlePaddle model (e.g., PP-OCRv5)");
            Console.WriteLine("2. Place it in one of the above directories");
            Console.WriteLine("3. Ensure it contains:");
            Console.WriteLine($"     - {modelJsonFile}");
            Console.WriteLine($"     - {modelParamsFile}");
            Console.WriteLine($"     - {configYamlFile} (optional)");
            Console.WriteLine("\nAlternatively, you can convert an existing ONNX model:");
            Console.WriteLine("  The converter can also work with ONNX models directly.");
            return;
        }

        var modelJsonPath = Path.Combine(paddleModelDir, modelJsonFile);
        var modelParamsPath = Path.Combine(paddleModelDir, modelParamsFile);
        var configYamlPath = Path.Combine(paddleModelDir, configYamlFile);

        // Some layouts place model files under a nested folder (e.g. PP-OCRv5_mobile_det_infer/PP-OCRv5_mobile_det_infer)
        if (!File.Exists(modelJsonPath) || !File.Exists(modelParamsPath))
        {
            var nestedDir = Directory.EnumerateDirectories(paddleModelDir)
                .FirstOrDefault(dir =>
                    File.Exists(Path.Combine(dir, modelJsonFile)) &&
                    File.Exists(Path.Combine(dir, modelParamsFile)));

            if (!string.IsNullOrWhiteSpace(nestedDir))
            {
                paddleModelDir = nestedDir;
                modelJsonPath = Path.Combine(paddleModelDir, modelJsonFile);
                modelParamsPath = Path.Combine(paddleModelDir, modelParamsFile);
                configYamlPath = Path.Combine(paddleModelDir, configYamlFile);
            }
        }

        // Validate model files
        if (!File.Exists(modelJsonPath))
        {
            Console.WriteLine($"✗ Model file not found: {modelJsonPath}");
            return;
        }

        if (!File.Exists(modelParamsPath))
        {
            Console.WriteLine($"✗ Parameters file not found: {modelParamsPath}");
            return;
        }

        try
        {
            Console.WriteLine($"\nModel directory: {paddleModelDir}");
            Console.WriteLine($"  - {modelJsonFile}: {new FileInfo(modelJsonPath).Length / 1024.0:F2} KB");
            Console.WriteLine($"  - {modelParamsFile}: {new FileInfo(modelParamsPath).Length / 1024.0:F2} KB");

            // Step 1: Check if paddle2onnx is installed
            Console.WriteLine("\n[Step 1] Checking paddle2onnx installation...");
            var converter = new Paddle2OnnxConverter();
            
            if (!converter.IsPaddle2OnnxInstalled())
            {
                Console.WriteLine("✗ paddle2onnx is not installed");
                Console.WriteLine(Paddle2OnnxConverter.GetInstallationInstructions());
                return;
            }
            Console.WriteLine("✓ paddle2onnx is installed");

            // Step 2: Convert PaddlePaddle model to ONNX (or use cached version)
            Console.WriteLine("\n[Step 2] Converting PaddlePaddle model to ONNX...");
            var sw = Stopwatch.StartNew();
            
            var onnxPath = converter.ConvertOrUseCache(
                modelJsonPath,
                modelParamsPath,
                opsetVersion: 11,
                forceReconvert: false
            );
            
            sw.Stop();
            Console.WriteLine($"✓ ONNX model ready: {onnxPath}");
            Console.WriteLine($"  Conversion time: {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  Model size: {new FileInfo(onnxPath).Length / 1024.0:F2} KB");

            // Step 3: Load preprocessing configuration (if available)
            PaddlePreprocessConfig? preprocessConfig = null;
            if (File.Exists(configYamlPath))
            {
                Console.WriteLine("\n[Step 3] Loading preprocessing configuration...");
                try
                {
                    preprocessConfig = PaddlePreprocessConfig.FromYaml(configYamlPath);
                    Console.WriteLine($"✓ Preprocessing config loaded:");
                    Console.WriteLine($"  {preprocessConfig}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠ Failed to load preprocessing config: {ex.Message}");
                    Console.WriteLine("  Continuing without preprocessing config...");
                }
            }
            else
            {
                Console.WriteLine("\n[Step 3] No preprocessing config file found, using defaults");
            }

            // Step 4: Create OpenVINO model
            Console.WriteLine("\n[Step 4] Loading ONNX model with OpenVINO...");
            var modelConfig = new OpenVinoModelConfig
            {
                ModelPath = onnxPath
            };

            using var model = new OpenVinoModel("paddle_det", modelConfig);
            Console.WriteLine("✓ OpenVINO model loaded");

            // Print model info
            var inputInfo = model.GetInputs();
            var outputInfo = model.GetOutputs();

            Console.WriteLine($"\n  Model inputs ({inputInfo.Length}):");
            foreach (var info in inputInfo)
            {
                var shapeStr = info.Shape != null ? $"[{string.Join(", ", info.Shape)}]" : "unknown";
                Console.WriteLine($"    - {info.Name}: {shapeStr}");
            }

            Console.WriteLine($"\n  Model outputs ({outputInfo.Length}):");
            foreach (var info in outputInfo)
            {
                var shapeStr = info.Shape != null ? $"[{string.Join(", ", info.Shape)}]" : "unknown";
                Console.WriteLine($"    - {info.Name}: {shapeStr}");
            }

            // Step 5: Create pipeline
            Console.WriteLine("\n[Step 5] Creating inference pipeline...");
            var pipelineConfig = new OpenVinoPipelineConfig
            {
                OpenVinoDevice = "CPU",
                OpenVinoPerformanceMode = "latency",
                OpenVinoNumStreams = 1,
                OpenVinoEnableMmap = true
            };

            using var pipeline = model.CreatePipeline("default", pipelineConfig);
            Console.WriteLine("✓ Pipeline created");

            // Step 6: Prepare test input
            Console.WriteLine("\n[Step 6] Preparing test input...");
            
            // Get input shape from model or preprocessing config
            int[] inputShape;
            if (inputInfo.Length > 0 && inputInfo[0].Shape != null)
            {
                var shape = inputInfo[0].Shape;
                inputShape = new int[shape.Length];
                for (int i = 0; i < shape.Length; i++)
                {
                    inputShape[i] = (int)shape[i];
                    // Handle dynamic batch dimension
                    if (i == 0 && (inputShape[i] <= 0 || inputShape[i] > 100))
                    {
                        inputShape[i] = 1;  // Set batch size to 1
                    }
                }
            }
            else if (preprocessConfig?.ImageShape != null)
            {
                inputShape = preprocessConfig.GetInputShape(batchSize: 1);
            }
            else
            {
                // Default shape for common OCR detection models: [1, 3, 640, 640]
                inputShape = new[] { 1, 3, 640, 640 };
                Console.WriteLine("  Using default input shape (no shape info available)");
            }

            Console.WriteLine($"  Input shape: [{string.Join(", ", inputShape)}]");

            // Create random test data
            var inputSize = 1;
            foreach (var dim in inputShape)
                inputSize *= dim;

            var random = new Random(42);
            var inputData = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                inputData[i] = (float)random.NextDouble();
            }

            // Apply preprocessing if configured
            if (preprocessConfig != null && inputShape.Length >= 3)
            {
                try
                {
                    var channels = preprocessConfig.IsCHW ? inputShape[1] : inputShape[3];
                    var height = preprocessConfig.IsCHW ? inputShape[2] : inputShape[1];
                    var width = preprocessConfig.IsCHW ? inputShape[3] : inputShape[2];
                    
                    inputData = preprocessConfig.Preprocess(inputData, channels, height, width);
                    Console.WriteLine("  ✓ Preprocessing applied");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ⚠ Preprocessing failed: {ex.Message}");
                }
            }

            // Step 7: Run inference
            Console.WriteLine("\n[Step 7] Running inference...");
            
            // Warmup
            Console.WriteLine("  Warming up...");
            _ = pipeline.Run(inputData, inputShape.Select(x => (long)x).ToArray());

            // Benchmark
            const int iterations = 10;
            sw.Restart();
            
            OpenVinoTensorOutput[]? outputs = null;
            for (int i = 0; i < iterations; i++)
            {
                outputs = pipeline.Run(inputData, inputShape.Select(x => (long)x).ToArray());
            }
            
            sw.Stop();
            var avgTime = sw.ElapsedMilliseconds / (double)iterations;

            Console.WriteLine($"✓ Inference completed");
            Console.WriteLine($"  Average inference time: {avgTime:F2}ms ({iterations} iterations)");
            Console.WriteLine($"  Throughput: {1000.0 / avgTime:F2} inferences/sec");

            // Step 8: Display outputs
            Console.WriteLine("\n[Step 8] Output results:");
            if (outputs != null)
            {
                Console.WriteLine($"  Number of outputs: {outputs.Length}");
                for (int i = 0; i < outputs.Length; i++)
                {
                    var output = outputs[i];
                    var shapeStr = string.Join(", ", output.Shape);
                    var dataSize = output.Data.Length;
                    
                    Console.WriteLine($"\n  Output {i}:");
                    Console.WriteLine($"    Shape: [{shapeStr}]");
                    Console.WriteLine($"    Size: {dataSize} elements");
                    Console.WriteLine($"    Data type: float32");
                    
                    // Show first few values
                    var previewCount = Math.Min(5, dataSize);
                    var preview = output.Data.Take(previewCount).Select(x => $"{x:F4}");
                    Console.WriteLine($"    First {previewCount} values: [{string.Join(", ", preview)}]");
                    
                    // Statistics
                    var min = output.Data.Min();
                    var max = output.Data.Max();
                    var mean = output.Data.Average();
                    Console.WriteLine($"    Stats: min={min:F4}, max={max:F4}, mean={mean:F4}");
                }
            }

            // Step 9: Cache information
            Console.WriteLine("\n[Step 9] Conversion cache info:");
            var cacheSize = converter.GetCacheSize();
            Console.WriteLine($"  Cache size: {cacheSize / 1024.0:F2} KB");
            Console.WriteLine($"  Cache can be cleared with: converter.ClearCache()");

            Console.WriteLine("\n✓ Case Paddle-OpenVINO 通过: All steps completed successfully!");
            Console.WriteLine("\nNext steps:");
            Console.WriteLine("  1. Replace random input with real image data");
            Console.WriteLine("  2. Implement post-processing for model outputs");
            Console.WriteLine("  3. Integrate into your application pipeline");

        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Case Paddle-OpenVINO 失败: {ex.Message}");
            Console.WriteLine($"\nStack trace:\n{ex.StackTrace}");
            
            if (ex.InnerException != null)
            {
                Console.WriteLine($"\nInner exception: {ex.InnerException.Message}");
            }
        }
    }
}
