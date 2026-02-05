using System;
using System.IO;
using System.Linq;
using CudaRS;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case1InputInfoTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 1] Model Input Information Query");
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

            var model = new OpenVinoModel("test_input_info", config);

            // Get inputs
            var inputs = model.GetInputs();

            Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
            Console.WriteLine($"Number of inputs: {inputs.Length}");
            Console.WriteLine();

            // Display each input
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                Console.WriteLine($"Input {i}:");
                Console.WriteLine($"  Name: {input.Name}");
                Console.WriteLine($"  Shape: [{string.Join(", ", input.Shape)}]");
                Console.WriteLine($"  ElementType: {input.ElementType}");
                Console.WriteLine();
            }

            // Validation for YOLO model
            if (inputs.Length > 0)
            {
                var firstInput = inputs[0];
                
                // Check if it's a typical YOLO input shape (batch, channels, height, width)
                if (firstInput.Shape.Length == 4)
                {
                    var batch = firstInput.Shape[0];
                    var channels = firstInput.Shape[1];
                    var height = firstInput.Shape[2];
                    var width = firstInput.Shape[3];
                    
                    Console.WriteLine("Input shape analysis:");
                    Console.WriteLine($"  Batch: {batch}");
                    Console.WriteLine($"  Channels: {channels}");
                    Console.WriteLine($"  Height: {height}");
                    Console.WriteLine($"  Width: {width}");
                    Console.WriteLine();

                    if (channels == 3 && height == width)
                    {
                        Console.WriteLine($"✓ Detected standard YOLO input: {channels} channels, {height}x{width}");
                    }
                }
            }

            // Clean up
            model.Dispose();

            Console.WriteLine("✓ Case 1 通过: Successfully queried model input information");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Case 1 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
