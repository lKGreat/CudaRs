using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Linq;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;
using CudaRS.OpenVino;

namespace CudaRS.Paddle;

/// <summary>
/// Converts PaddleOCR det/rec models to OpenVINO IR via the native SDK.
/// </summary>
public sealed class PaddleToIrConverter
{
    private const int DefaultBufferSize = 1024;
    private readonly string _assetsDir;
    private readonly string _fallbackOutputDir;
    private readonly Paddle2OnnxConverter? _onnx;
    private readonly OpenVinoIrConverter? _ir;

    public PaddleToIrConverter(
        string? assetsDir = null,
        string? pythonPath = null,
        string? onnxCacheDir = null,
        string? irCacheDir = null,
        bool preferPython = true)
    {
        _assetsDir = assetsDir ?? AppContext.BaseDirectory;
        _fallbackOutputDir = irCacheDir ?? Path.Combine(Path.GetTempPath(), "paddle_ocr_ir_cache");

        if (preferPython)
        {
            try
            {
                _onnx = new Paddle2OnnxConverter(pythonPath, onnxCacheDir);
                _ir = new OpenVinoIrConverter(pythonPath, irCacheDir);
            }
            catch
            {
                _onnx = null;
                _ir = null;
            }
        }
    }

    public bool IsReady()
    {
        return IsPythonReady() || HasAssetsAt(_assetsDir) ||
               HasAssetsAt(Path.Combine(_assetsDir, "runtimes", GetRuntimeIdentifier(), "native"));
    }

    public (string DetXml, string RecXml) ConvertOrUseCache(
        string detModelDir,
        string recModelDir,
        int opsetVersion = 11,
        bool forceReconvert = false,
        bool compressToFp16 = false)
    {
        if (IsPythonReady())
        {
            var detDir = ResolveModelDir(detModelDir);
            var recDir = ResolveModelDir(recModelDir);

            var detOnnx = _onnx!.ConvertDirectory(detDir, outputPath: null, opsetVersion: opsetVersion);
            var recOnnx = _onnx!.ConvertDirectory(recDir, outputPath: null, opsetVersion: opsetVersion);

            var detXml = _ir!.ConvertOrUseCache(detOnnx, forceReconvert, compressToFp16);
            var recXml = _ir!.ConvertOrUseCache(recOnnx, forceReconvert, compressToFp16);

            return (detXml, recXml);
        }

        var options = new PaddleOcrIrOptions
        {
            OpsetVersion = opsetVersion,
            CompressToFp16 = compressToFp16,
            ForceReconvert = forceReconvert,
            CacheDir = _fallbackOutputDir
        };

        return Convert(detModelDir, recModelDir, _fallbackOutputDir, options);
    }

    public (string DetXml, string RecXml) Convert(
        string detModelDir,
        string recModelDir,
        string outputDir,
        PaddleOcrIrOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(detModelDir))
            throw new ArgumentException("detModelDir is required", nameof(detModelDir));
        if (string.IsNullOrWhiteSpace(recModelDir))
            throw new ArgumentException("recModelDir is required", nameof(recModelDir));
        if (string.IsNullOrWhiteSpace(outputDir))
            throw new ArgumentException("outputDir is required", nameof(outputDir));

        var detDir = ResolveModelDir(detModelDir);
        var recDir = ResolveModelDir(recModelDir);

        Directory.CreateDirectory(outputDir);

        var opts = options ?? new PaddleOcrIrOptions();
        var optionsJson = JsonSerializer.Serialize(opts, PaddleOcrIrOptions.JsonOptions);

        var detBytes = Encoding.UTF8.GetBytes(detDir);
        var recBytes = Encoding.UTF8.GetBytes(recDir);
        var outBytes = Encoding.UTF8.GetBytes(outputDir);
        var optBytes = Encoding.UTF8.GetBytes(optionsJson);

        byte[] detBuf = new byte[DefaultBufferSize];
        byte[] recBuf = new byte[DefaultBufferSize];

        for (var attempt = 0; attempt < 3; attempt++)
        {
            unsafe
            {
                fixed (byte* detPtr = detBytes)
                fixed (byte* recPtr = recBytes)
                fixed (byte* outPtr = outBytes)
                fixed (byte* optPtr = optBytes)
                fixed (byte* detOut = detBuf)
                fixed (byte* recOut = recBuf)
                {
                    var err = SdkNative.ConvertPaddleOcrToIr(
                        detPtr, (nuint)detBytes.Length,
                        recPtr, (nuint)recBytes.Length,
                        outPtr, (nuint)outBytes.Length,
                        optPtr, (nuint)optBytes.Length,
                        detOut, (nuint)detBuf.Length, out var detWritten,
                        recOut, (nuint)recBuf.Length, out var recWritten);

                    if (err == SdkErr.OutOfMemory)
                    {
                        var neededDet = (int)detWritten;
                        var neededRec = (int)recWritten;
                        var resized = false;
                        if (neededDet > detBuf.Length)
                        {
                            detBuf = new byte[Math.Max(neededDet, detBuf.Length * 2)];
                            resized = true;
                        }
                        if (neededRec > recBuf.Length)
                        {
                            recBuf = new byte[Math.Max(neededRec, recBuf.Length * 2)];
                            resized = true;
                        }
                        if (!resized)
                            break;
                        continue;
                    }

                    SdkCheck.ThrowIfError(err);

                    var detXml = Encoding.UTF8.GetString(detBuf, 0, (int)detWritten);
                    var recXml = Encoding.UTF8.GetString(recBuf, 0, (int)recWritten);
                    return (detXml, recXml);
                }
            }
        }

        throw new InvalidOperationException("Failed to resize output buffers for conversion.");
    }

    public static string GetInstallationInstructions()
        => "Python-based conversion:\n" +
           Paddle2OnnxConverter.GetInstallationInstructions() + "\n" +
           OpenVinoIrConverter.GetInstallationInstructions() + "\n" +
           "Native conversion:\n" +
           "Requires the bundled Python runtime and scripts packaged with CudaRS.Native.";

    private bool IsPythonReady()
        => _onnx != null && _ir != null && _onnx.IsPaddle2OnnxInstalled() && _ir.IsOpenVinoInstalled();

    private static string ResolveModelDir(string modelDir)
    {
        if (string.IsNullOrWhiteSpace(modelDir) || !Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        var modelJson = Path.Combine(modelDir, "inference.json");
        var modelParams = Path.Combine(modelDir, "inference.pdiparams");
        if (File.Exists(modelJson) && File.Exists(modelParams))
            return modelDir;

        var nested = Directory.EnumerateDirectories(modelDir)
            .FirstOrDefault(dir =>
                File.Exists(Path.Combine(dir, "inference.json")) &&
                File.Exists(Path.Combine(dir, "inference.pdiparams")));

        if (!string.IsNullOrEmpty(nested))
            return nested;

        throw new FileNotFoundException($"Paddle model files not found under: {modelDir}");
    }

    private static bool HasAssetsAt(string root)
    {
        var pythonDir = Path.Combine(root, "python");
        var scriptsDir = Path.Combine(root, "scripts");
        return Directory.Exists(pythonDir) && Directory.Exists(scriptsDir);
    }

    private static string GetRuntimeIdentifier()
    {
        var arch = RuntimeInformation.OSArchitecture;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return arch == Architecture.Arm64 ? "win-arm64" : "win-x64";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return arch == Architecture.Arm64 ? "osx-arm64" : "osx-x64";
        return arch == Architecture.Arm64 ? "linux-arm64" : "linux-x64";
    }
}

public sealed class PaddleOcrIrOptions
{
    [JsonPropertyName("opset_version")]
    public int OpsetVersion { get; set; } = 11;

    [JsonPropertyName("compress_to_fp16")]
    public bool CompressToFp16 { get; set; }

    [JsonPropertyName("enable_validation")]
    public bool EnableValidation { get; set; } = true;

    [JsonPropertyName("force_reconvert")]
    public bool ForceReconvert { get; set; }

    [JsonPropertyName("timeout_secs")]
    public int TimeoutSeconds { get; set; } = 300;

    [JsonPropertyName("cache_dir")]
    public string? CacheDir { get; set; }

    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = null,
        WriteIndented = false
    };
}
