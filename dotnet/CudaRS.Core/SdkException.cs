using System;
using System.Text;
using CudaRS.Native;

namespace CudaRS.Core;

public sealed class SdkException : Exception
{
    public SdkException(SdkErr errorCode, string? message)
        : base(message ?? errorCode.ToString())
    {
        ErrorCode = errorCode;
    }

    public SdkException(SdkErr errorCode, string? message, string? missingFile, string[]? searchedPaths, string? suggestion)
        : base(message ?? errorCode.ToString())
    {
        ErrorCode = errorCode;
        MissingFile = missingFile;
        SearchedPaths = searchedPaths;
        Suggestion = suggestion;
    }

    public SdkErr ErrorCode { get; }
    public string? MissingFile { get; }
    public string[]? SearchedPaths { get; }
    public string? Suggestion { get; }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"SDK错误: {ErrorCode} - {Message}");
        
        if (!string.IsNullOrEmpty(MissingFile))
            sb.AppendLine($"缺失文件: {MissingFile}");
        
        if (SearchedPaths?.Length > 0)
        {
            sb.AppendLine("已搜索路径:");
            foreach (var path in SearchedPaths)
                sb.AppendLine($"  - {path}");
        }
        
        if (!string.IsNullOrEmpty(Suggestion))
            sb.AppendLine($"解决建议: {Suggestion}");
        
        return sb.ToString();
    }
}
