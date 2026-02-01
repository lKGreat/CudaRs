using CudaRS;
using CudaRS.Core;

Console.WriteLine("=== CudaRS Example ===\n");

// Get library version
Console.WriteLine($"CudaRS Version: {Cuda.Version}");

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
        try
        {
            using var cublasHandle = new CublasHandle();
            Console.WriteLine($"cuBLAS Version: {cublasHandle.Version}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"cuBLAS: {ex.Message}");
        }

        try
        {
            Console.WriteLine($"cuDNN Version: {CudnnHandle.Version}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"cuDNN: {ex.Message}");
        }

        try
        {
            var (major, minor) = Nvrtc.Version;
            Console.WriteLine($"NVRTC Version: {major}.{minor}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"NVRTC: {ex.Message}");
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

Console.WriteLine("\n=== Example Complete ===");
