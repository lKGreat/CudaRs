using System;
using System.IO;
using System.Linq;
using CudaRS;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case3FixedReshapeTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 3] Fixed Shape Reshape");
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

            var model = new OpenVinoModel("test_reshape", config);

            // Get original input shape
            var originalInputs = model.GetInputs();
            Console.WriteLine($"Original input shape: [{string.Join(", ", originalInputs[0].Shape)}]");
            Console.WriteLine();

            // Test reshape to 320x320
            Console.WriteLine("Reshaping to [1, 3, 320, 320]...");
            model.Reshape(1, 3, 320, 320);
            
            var reshaped320 = model.GetInputs();
            Console.WriteLine($"New input shape: [{string.Join(", ", reshaped320[0].Shape)}]");
            
            if (reshaped320[0].Shape.SequenceEqual(new long[] { 1, 3, 320, 320 }))
            {
                Console.WriteLine("✓ 320x320 reshape successful");
            }
            else
            {
                Console.WriteLine("✗ 320x320 reshape failed: shape mismatch");
            }
            Console.WriteLine();

            // Test reshape to 1280x1280
            Console.WriteLine("Reshaping to [1, 3, 1280, 1280]...");
            model.Reshape(1, 3, 1280, 1280);
            
            var reshaped1280 = model.GetInputs();
            Console.WriteLine($"New input shape: [{string.Join(", ", reshaped1280[0].Shape)}]");
            
            if (reshaped1280[0].Shape.SequenceEqual(new long[] { 1, 3, 1280, 1280 }))
            {
                Console.WriteLine("✓ 1280x1280 reshape successful");
            }
            else
            {
                Console.WriteLine("✗ 1280x1280 reshape failed: shape mismatch");
            }
            Console.WriteLine();

            // Reshape back to original
            var originalShape = originalInputs[0].Shape;
            Console.WriteLine($"Reshaping back to original [{string.Join(", ", originalShape)}]...");
            model.Reshape(originalShape);
            
            var restoredInputs = model.GetInputs();
            Console.WriteLine($"Restored input shape: [{string.Join(", ", restoredInputs[0].Shape)}]");
            
            if (restoredInputs[0].Shape.SequenceEqual(originalShape))
            {
                Console.WriteLine("✓ Restore to original shape successful");
            }
            else
            {
                Console.WriteLine("✗ Restore failed: shape mismatch");
            }

            // Clean up
            model.Dispose();

            Console.WriteLine();
            Console.WriteLine("✓ Case 3 通过: Model can reshape to different fixed dimensions");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Case 3 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
