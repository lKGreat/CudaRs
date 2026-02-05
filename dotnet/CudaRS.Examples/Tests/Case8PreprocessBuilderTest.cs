using System;
using System.IO;
using CudaRS.OpenVino;
using CudaRS.OpenVino.Preprocessing;

namespace CudaRS.Examples.Tests;

public static class Case8PreprocessBuilderTest
{
    public static void Run()
    {
        Console.WriteLine("\n[Case 8] PreprocessBuilder Implementation");
        Console.WriteLine("==========================================");

        var modelPath = Config.TestYoloModel;
        
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"⚠ Test model not found: {modelPath}");
            Console.WriteLine($"  Please set Config.TestYoloModel to a valid ONNX model path");
            return;
        }

        try
        {
            // Load original model
            var config = new OpenVinoModelConfig
            {
                ModelPath = modelPath,
            };

            using var originalModel = new OpenVinoModel("test_preprocess", config);
            Console.WriteLine($"✓ Loaded original model: {Path.GetFileName(modelPath)}");

            // Get original input info
            var originalInputs = originalModel.GetInputs();
            if (originalInputs.Length > 0)
            {
                var input = originalInputs[0];
                Console.WriteLine($"  Original input: {input.Name}");
                Console.WriteLine($"  Shape: [{string.Join(", ", input.Shape)}]");
                Console.WriteLine($"  ElementType: {input.ElementType}");
            }

            // Create preprocessing builder
            Console.WriteLine("\nConfiguring preprocessing...");
            using var builder = PreprocessBuilder.Create(originalModel);
            
            // Configure preprocessing: U8 input -> NCHW model with linear resize
            var preprocessedModel = builder
                .Input()
                    .TensorFormat(TensorElementType.U8, "NHWC")
                    .ModelLayout("NCHW")
                    .Resize(ResizeAlgorithm.Linear)
                .Build();

            Console.WriteLine("✓ Built preprocessed model");

            // Verify the preprocessed model can be used
            using (preprocessedModel)
            {
                var preprocessedInputs = preprocessedModel.GetInputs();
                if (preprocessedInputs.Length > 0)
                {
                    var input = preprocessedInputs[0];
                    Console.WriteLine($"\nPreprocessed model input:");
                    Console.WriteLine($"  Name: {input.Name}");
                    Console.WriteLine($"  Shape: [{string.Join(", ", input.Shape)}]");
                    Console.WriteLine($"  ElementType: {input.ElementType}");
                }

                // Try to create a pipeline with the preprocessed model
                var pipelineConfig = new OpenVinoPipelineConfig();
                using var pipeline = preprocessedModel.CreatePipeline("test_pipeline", pipelineConfig);
                Console.WriteLine("✓ Created pipeline from preprocessed model");
            }

            Console.WriteLine("\n✓ Case 8 通过: PreprocessBuilder working correctly");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Case 8 失败: {ex.Message}");
            Console.WriteLine($"  Stack trace: {ex.StackTrace}");
        }
    }
}
