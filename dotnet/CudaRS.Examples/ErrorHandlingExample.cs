using System;
using CudaRS.Core;
using CudaRS.Native;
using CudaRS.Yolo;

namespace CudaRS.Examples;

/// <summary>
/// 演示增强的错误处理功能
/// </summary>
public static class ErrorHandlingExample
{
    public static void DemonstrateErrorHandling()
    {
        Console.WriteLine("=== Error Handling Demo ===\n");

        // 示例1: 捕获详细的依赖缺失错误
        Console.WriteLine("Example 1: Dependency Missing Error");
        try
        {
            // 尝试加载不存在的模型
            var pipeline = CudaRsFluent.Create()
                .Pipeline()
                .ForYolo("/path/to/nonexistent/model.onnx", cfg =>
                {
                    cfg.Version = YoloVersion.V8;
                    cfg.Task = YoloTask.Detect;
                })
                .AsOpenVino()
                .BuildYolo();
        }
        catch (SdkException ex)
        {
            Console.WriteLine($"Caught SdkException:");
            Console.WriteLine($"  Error Code: {ex.ErrorCode}");
            Console.WriteLine($"  Message: {ex.Message}");
            
            if (ex.MissingFile != null)
                Console.WriteLine($"  Missing File: {ex.MissingFile}");
            
            if (ex.SearchedPaths != null && ex.SearchedPaths.Length > 0)
            {
                Console.WriteLine($"  Searched Paths:");
                foreach (var path in ex.SearchedPaths)
                    Console.WriteLine($"    - {path}");
            }
            
            if (ex.Suggestion != null)
                Console.WriteLine($"  Suggestion: {ex.Suggestion}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Caught general exception: {ex.Message}");
        }

        Console.WriteLine();

        // 示例2: 错误码检查
        Console.WriteLine("Example 2: Error Code Types");
        var errorCodes = new[]
        {
            SdkErr.Ok,
            SdkErr.InvalidArg,
            SdkErr.MissingDependency,
            SdkErr.DllNotFound,
            SdkErr.ModelLoadFailed,
            SdkErr.ConfigInvalid,
        };

        foreach (var code in errorCodes)
        {
            Console.WriteLine($"  {code,20}: {GetErrorDescription(code)}");
        }

        Console.WriteLine();

        // 示例3: 使用详细错误信息进行调试
        Console.WriteLine("Example 3: Debugging with Detailed Errors");
        Console.WriteLine("  Set CUDARS_DIAG=1 environment variable for diagnostic output");
        Console.WriteLine("  Detailed errors include:");
        Console.WriteLine("    - Missing dependency files");
        Console.WriteLine("    - All searched paths");
        Console.WriteLine("    - Configuration suggestions");
        Console.WriteLine("    - Environment variable hints");
    }

    private static string GetErrorDescription(SdkErr err) => err switch
    {
        SdkErr.Ok => "操作成功",
        SdkErr.InvalidArg => "无效参数",
        SdkErr.OutOfMemory => "内存不足",
        SdkErr.Runtime => "运行时错误",
        SdkErr.Unsupported => "不支持的操作",
        SdkErr.NotFound => "未找到资源",
        SdkErr.Timeout => "操作超时",
        SdkErr.Busy => "资源忙",
        SdkErr.Io => "IO错误",
        SdkErr.Permission => "权限不足",
        SdkErr.Canceled => "操作已取消",
        SdkErr.BadState => "状态错误",
        SdkErr.VersionMismatch => "版本不匹配",
        SdkErr.Backend => "后端错误",
        SdkErr.MissingDependency => "缺少依赖",
        SdkErr.DllNotFound => "DLL未找到",
        SdkErr.ModelLoadFailed => "模型加载失败",
        SdkErr.ConfigInvalid => "配置无效",
        _ => "未知错误"
    };
}
