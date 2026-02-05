using System;
using System.IO;
using System.Linq;
using CudaRS;
using CudaRS.OpenVino;

namespace CudaRS.Examples.Tests;

public static class Case2OutputInfoTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 2] Model Output Information Query");
        Console.WriteLine("=========================================");

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

            var model = new OpenVinoModel("test_output_info", config);

            // Get outputs
            var outputs = model.GetOutputs();

            Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
            Console.WriteLine($"Number of outputs: {outputs.Length}");
            Console.WriteLine();

            // Display each output
            for (int i = 0; i < outputs.Length; i++)
            {
                var output = outputs[i];
                Console.WriteLine($"Output {i}:");
                Console.WriteLine($"  Name: {output.Name}");
                Console.WriteLine($"  Shape: [{string.Join(", ", output.Shape)}]");
                Console.WriteLine($"  ElementType: {output.ElementType}");
                Console.WriteLine();
            }

            // Validation for YOLO model
            if (outputs.Length > 0)
            {
                Console.WriteLine($"Output count validation:");
                Console.WriteLine($"  Expected for YOLOv11: 3 outputs (detection heads)");
                Console.WriteLine($"  Actual: {outputs.Length} outputs");
                
                if (outputs.Length == 3)
                {
                    Console.WriteLine("  ✓ Output count matches YOLOv11 expected");
                }
                else if (outputs.Length == 1)
                {
                    Console.WriteLine("  ✓ Single output (combined detection head)");
                }
                Console.WriteLine();
            }

            // Analyze first output shape
            if (outputs.Length > 0)
            {
                var firstOutput = outputs[0];
                Console.WriteLine("First output analysis:");
                Console.WriteLine($"  Shape: [{string.Join(", ", firstOutput.Shape)}]");
                
                if (firstOutput.Shape.Length >= 2)
                {
                    var lastDim = firstOutput.Shape[firstOutput.Shape.Length - 1];
                    Console.WriteLine($"  Last dimension: {lastDim}");
                    
                    if (lastDim == 84)
                    {
                        Console.WriteLine("  ✓ Detected YOLO detection format (80 classes + 4 bbox coords)");
                    }
                    else if (lastDim > 84)
                    {
                        Console.WriteLine($"  ✓ Extended YOLO format ({lastDim} values per detection)");
                    }
                }
                Console.WriteLine();
            }

            // Clean up
            model.Dispose();

            Console.WriteLine("✓ Case 2 通过: Successfully queried model output information");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Case 2 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
