using CudaRS;
using CudaRS.Core;
using System.Runtime.InteropServices;

Console.WriteLine("=== CudaRS Example ===\n");

// Get library version
Console.WriteLine($"CudaRS Version: {Cuda.Version}");

static bool HasExport(string name)
{
    if (!NativeLibrary.TryLoad("cudars_ffi", out var handle))
        return false;
    try
    {
        return NativeLibrary.TryGetExport(handle, name, out _);
    }
    finally
    {
        NativeLibrary.Free(handle);
    }
}

// Check CUDA devices
try
{
    var deviceCount = Cuda.DeviceCount;
    Console.WriteLine($"CUDA Devices Found: {deviceCount}");

    if (deviceCount > 0)
    {
        // Get driver version
        var driverVersion = CudaDriver.Version;
        Console.WriteLine($"CUDA Driver Version: {driverVersion / 1000}.{(driverVersion % 1000) / 10}");

        // Set device 0
        Cuda.CurrentDevice = 0;
        Console.WriteLine($"Using Device: {Cuda.CurrentDevice}");

        // Test memory allocation
        Console.WriteLine("\n--- Memory Test ---");
        const int arraySize = 1024;
        var hostData = new float[arraySize];
        for (int i = 0; i < arraySize; i++)
            hostData[i] = i * 0.5f;

        using var deviceBuffer = hostData.ToDevice();
        Console.WriteLine($"Allocated {arraySize} floats on device");

        // Copy back and verify
        var resultData = deviceBuffer.ToArray();
        var correct = true;
        for (int i = 0; i < arraySize && correct; i++)
            correct = Math.Abs(hostData[i] - resultData[i]) < 0.0001f;
        Console.WriteLine($"Data verification: {(correct ? "PASSED" : "FAILED")}");

        // Test streams and events
        Console.WriteLine("\n--- Stream & Event Test ---");
        using var stream = new CudaStream();
        using var startEvent = new CudaEvent();
        using var endEvent = new CudaEvent();

        startEvent.Record(stream);
        // Simulate some work
        Thread.Sleep(10);
        endEvent.Record(stream);
        stream.Synchronize();

        var elapsedMs = endEvent.ElapsedTime(startEvent);
        Console.WriteLine($"Elapsed time: {elapsedMs:F3} ms");

        // Test library handles
        Console.WriteLine("\n--- Library Handles Test ---");
        if (HasExport("cudars_cublas_create"))
        {
            try
            {
                using var cublasHandle = new CublasHandle();
                Console.WriteLine($"cuBLAS Version: {cublasHandle.Version}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"cuBLAS: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("cuBLAS: not available");
        }

        if (HasExport("cudars_cudnn_get_version"))
        {
            try
            {
                Console.WriteLine($"cuDNN Version: {CudnnHandle.Version}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"cuDNN: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("cuDNN: not available");
        }

        if (HasExport("cudars_nvrtc_version"))
        {
            try
            {
                var (major, minor) = Nvrtc.Version;
                Console.WriteLine($"NVRTC Version: {major}.{minor}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"NVRTC: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("NVRTC: not available");
        }

        // Synchronize device
        Cuda.Synchronize();
    }
}
catch (CudaException ex)
{
    Console.WriteLine($"CUDA Error: {ex.ErrorCode} - {ex.Message}");
}

// GPU Management via NVML
Console.WriteLine("\n--- GPU Management (NVML) ---");
if (HasExport("cudars_nvml_init"))
{
    try
    {
        var gpuCount = GpuManagement.DeviceCount;
        Console.WriteLine($"GPUs Found: {gpuCount}");

        for (uint i = 0; i < gpuCount; i++)
        {
            Console.WriteLine($"\nGPU {i}:");
            
            var memInfo = GpuManagement.GetMemoryInfo(i);
            Console.WriteLine($"  Memory: {memInfo.Used / (1024 * 1024)} MB / {memInfo.Total / (1024 * 1024)} MB");

            var utilRates = GpuManagement.GetUtilizationRates(i);
            Console.WriteLine($"  GPU Utilization: {utilRates.Gpu}%");
            Console.WriteLine($"  Memory Utilization: {utilRates.Memory}%");

            var temp = GpuManagement.GetTemperature(i);
            Console.WriteLine($"  Temperature: {temp}°C");

            try
            {
                var power = GpuManagement.GetPowerUsage(i);
                Console.WriteLine($"  Power: {power / 1000.0:F1} W");
            }
            catch { /* Power reading not available */ }

            try
            {
                var fan = GpuManagement.GetFanSpeed(i);
                Console.WriteLine($"  Fan Speed: {fan}%");
            }
            catch { /* Fan speed not available */ }
        }

        GpuManagement.Shutdown();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"NVML Error: {ex.Message}");
    }
}
else
{
    Console.WriteLine("NVML: not available");
}

Console.WriteLine("\n=== Example Complete ===");

// Fluent demo layer
Console.WriteLine("\n=== Fluent Pipeline Demo ===\n");

var pipeline = InferencePipeline.Create()
    .WithName("SecurityMultiChannel")
    .WithChannel("cam-1", channel => channel
        .WithShape(1920, 1080, 3)
        .WithDataType("uint8")
        .WithScenePriority(SceneLevel.L2)
        .WithFpsRange(5, 25)
        .WithBatching(1, 0))
    .WithChannel("cam-2", channel => channel
        .WithShape(1280, 720, 3)
        .WithDataType("uint8")
        .WithScenePriority(SceneLevel.L1)
        .WithFpsRange(5, 20)
        .WithBatching(1, 0))
    .WithModel("detector", model => model
        .FromPath("models/detector.onnx")
        .WithBackend("onnx")
        .OnDevice("cuda:0")
        .WithPrecision("fp16"))
    .WithPreprocessStage("resize+normalize")
    .WithInferStage("detector")
    .WithPostprocessStage("nms")
    .WithExecution(options => options
        .WithMaxConcurrency(2)
        .WithStreamMode(StreamMode.Async)
        .WithStreamPoolSize(32)
        .WithMaxQueueDepth(5000))
    .Build();

var demoInputs = new Dictionary<string, ChannelInput>
{
    ["cam-1"] = new ChannelInput(new byte[0]) { SceneLevel = SceneLevel.L2 },
    ["cam-2"] = new ChannelInput(new byte[0]) { SceneLevel = SceneLevel.L1 },
};

var demoResult = pipeline.Run(new PipelineInput(demoInputs));
Console.WriteLine($"Pipeline: {demoResult.PipelineName}");
Console.WriteLine($"Success: {demoResult.Success}");
Console.WriteLine($"Elapsed: {demoResult.Elapsed.TotalMilliseconds:F3} ms");
if (demoResult.Diagnostics.Count > 0)
{
    Console.WriteLine("Diagnostics:");
    foreach (var message in demoResult.Diagnostics)
        Console.WriteLine($"- {message}");
}
