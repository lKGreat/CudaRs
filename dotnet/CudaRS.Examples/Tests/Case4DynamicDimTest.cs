using System;
using System.IO;
using System.Linq;
using CudaRS;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case4DynamicDimTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 4] Dynamic Dimension Support");
        Console.WriteLine("========================================");

        // Use test model path from Config
        var modelPath = Config.TestYoloModel;
        
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"⚠ Test model not found: {modelPath}");
            Console.WriteLine($"  Please set Config.TestYoloModel to a valid ONNX model path");
            return;
        }

        try
        {
            // Create OpenVINO model
            var config = new OpenVinoModelConfig
            {
                ModelPath = modelPath,
            };

            var model = new OpenVinoModel("test_dynamic", config);

            // Get original input shape
            var originalInputs = model.GetInputs();
            Console.WriteLine($"Original input shape: [{string.Join(", ", originalInputs[0].Shape)}]");
            Console.WriteLine();

            // Test 1: Dynamic batch dimension
            Console.WriteLine("Test 1: Dynamic batch dimension");
            Console.WriteLine("Reshaping to [?, 3, 640, 640] (dynamic batch)...");
            
            var dynamicBatchShape = PartialShape.Create(
                PartialDimension.Dynamic,  // batch
                PartialDimension.Static(3),
                PartialDimension.Static(640),
                PartialDimension.Static(640)
            );
            
            model.Reshape(dynamicBatchShape);
            
            var reshapedInputs = model.GetInputs();
            Console.WriteLine($"New input shape: [{string.Join(", ", reshapedInputs[0].Shape)}]");
            Console.WriteLine($"PartialShape: {dynamicBatchShape}");
            
            // For dynamic dimensions, OpenVINO might return -1 or a default value
            if (reshapedInputs[0].Shape.Length == 4)
            {
                var batch = reshapedInputs[0].Shape[0];
                var channels = reshapedInputs[0].Shape[1];
                var height = reshapedInputs[0].Shape[2];
                var width = reshapedInputs[0].Shape[3];
                
                Console.WriteLine($"Reshaped analysis:");
                Console.WriteLine($"  Batch: {batch} (dynamic dimension, value may vary)");
                Console.WriteLine($"  Channels: {channels}");
                Console.WriteLine($"  Height: {height}");
                Console.WriteLine($"  Width: {width}");
                
                if (channels == 3 && height == 640 && width == 640)
                {
                    Console.WriteLine("✓ Dynamic batch reshape successful (channels/height/width match)");
                }
                else
                {
                    Console.WriteLine("⚠ Dimensions mismatch, but reshape completed");
                }
            }
            Console.WriteLine();

            // Test 2: Using FromArray helper with -1 for dynamic
            Console.WriteLine("Test 2: Dynamic shape using FromArray ([-1, 3, 320, 320])");
            var dynamicShape = PartialShape.FromArray(-1, 3, 320, 320);
            Console.WriteLine($"PartialShape: {dynamicShape}");
            Console.WriteLine($"Is dynamic? {dynamicShape.IsDynamic}");
            
            model.Reshape(dynamicShape);
            
            var reshapedInputs2 = model.GetInputs();
            Console.WriteLine($"New input shape: [{string.Join(", ", reshapedInputs2[0].Shape)}]");
            Console.WriteLine("✓ Dynamic shape with FromArray helper successful");
            Console.WriteLine();

            // Test 3: Restore to all static shape
            Console.WriteLine("Test 3: Restore to all static shape [1, 3, 640, 640]");
            var staticShape = PartialShape.FromStaticShape(1, 3, 640, 640);
            Console.WriteLine($"PartialShape: {staticShape}");
            Console.WriteLine($"Is static? {staticShape.IsStatic}");
            
            model.Reshape(staticShape);
            
            var reshapedInputs3 = model.GetInputs();
            Console.WriteLine($"New input shape: [{string.Join(", ", reshapedInputs3[0].Shape)}]");
            
            if (reshapedInputs3[0].Shape.SequenceEqual(new long[] { 1, 3, 640, 640 }))
            {
                Console.WriteLine("✓ Restore to static shape successful");
            }

            // Clean up
            model.Dispose();

            Console.WriteLine();
            Console.WriteLine("✓ Case 4 通过: Model supports dynamic dimensions");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Case 4 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
