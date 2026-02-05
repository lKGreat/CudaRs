using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace CudaRS.Paddle;

/// <summary>
/// Converts PaddlePaddle models to ONNX format using paddle2onnx
/// </summary>
public sealed class Paddle2OnnxConverter
{
    private readonly string _pythonPath;
    private readonly string _converterScriptPath;
    private readonly string _cacheDir;

    public Paddle2OnnxConverter(string? pythonPath = null, string? cacheDir = null)
    {
        _pythonPath = pythonPath ?? "python";
        _cacheDir = cacheDir ?? Path.Combine(Path.GetTempPath(), "paddle2onnx_cache");
        
        // Find converter script
        var scriptName = "paddle2onnx_converter.py";
        var possiblePaths = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "scripts", scriptName),
            Path.Combine(Directory.GetCurrentDirectory(), "scripts", scriptName),
            Path.Combine(Directory.GetCurrentDirectory(), "..", "scripts", scriptName),
            Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "scripts", scriptName),
        };

        _converterScriptPath = string.Empty;
        foreach (var path in possiblePaths)
        {
            var fullPath = Path.GetFullPath(path);
            if (File.Exists(fullPath))
            {
                _converterScriptPath = fullPath;
                break;
            }
        }

        if (string.IsNullOrEmpty(_converterScriptPath))
        {
            throw new FileNotFoundException(
                $"Converter script '{scriptName}' not found. Searched paths: {string.Join(", ", possiblePaths)}");
        }

        // Ensure cache directory exists
        if (!Directory.Exists(_cacheDir))
        {
            Directory.CreateDirectory(_cacheDir);
        }
    }

    /// <summary>
    /// Convert PaddlePaddle model to ONNX, or use cached version if available
    /// </summary>
    public string ConvertOrUseCache(string modelJsonPath, string modelParamsPath, 
                                   int opsetVersion = 11, bool forceReconvert = false)
    {
        if (!File.Exists(modelJsonPath))
            throw new FileNotFoundException($"Model JSON file not found: {modelJsonPath}");
        if (!File.Exists(modelParamsPath))
            throw new FileNotFoundException($"Parameters file not found: {modelParamsPath}");

        // Generate cache key based on file paths and modification times
        var jsonInfo = new FileInfo(modelJsonPath);
        var paramsInfo = new FileInfo(modelParamsPath);
        var cacheKey = $"{Path.GetFileName(modelJsonPath)}_{jsonInfo.LastWriteTimeUtc:yyyyMMddHHmmss}_{paramsInfo.LastWriteTimeUtc:yyyyMMddHHmmss}_v{opsetVersion}";
        var cacheFilePath = Path.Combine(_cacheDir, $"{cacheKey}.onnx");

        // Use cached version if available and not forcing reconversion
        if (!forceReconvert && File.Exists(cacheFilePath))
        {
            Console.WriteLine($"Using cached ONNX model: {cacheFilePath}");
            return cacheFilePath;
        }

        // Convert model
        Console.WriteLine($"Converting PaddlePaddle model to ONNX...");
        Convert(Path.GetDirectoryName(modelJsonPath)!, cacheFilePath, 
               Path.GetFileName(modelJsonPath), Path.GetFileName(modelParamsPath), 
               opsetVersion);

        return cacheFilePath;
    }

    /// <summary>
    /// Convert PaddlePaddle model directory to ONNX
    /// </summary>
    public string ConvertDirectory(string modelDir, string? outputPath = null,
                                  string modelFilename = "inference.json",
                                  string paramsFilename = "inference.pdiparams",
                                  int opsetVersion = 11)
    {
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        var modelPath = Path.Combine(modelDir, modelFilename);
        var paramsPath = Path.Combine(modelDir, paramsFilename);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        if (!File.Exists(paramsPath))
            throw new FileNotFoundException($"Parameters file not found: {paramsPath}");

        // Use cache if no output path specified
        if (string.IsNullOrEmpty(outputPath))
        {
            return ConvertOrUseCache(modelPath, paramsPath, opsetVersion);
        }

        // Convert to specified output path
        Convert(modelDir, outputPath, modelFilename, paramsFilename, opsetVersion);
        return outputPath;
    }

    /// <summary>
    /// Convert PaddlePaddle model to ONNX
    /// </summary>
    public void Convert(string modelDir, string outputPath,
                       string modelFilename = "inference.json",
                       string paramsFilename = "inference.pdiparams",
                       int opsetVersion = 11,
                       bool enableValidation = true)
    {
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir}");

        var outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
        {
            Directory.CreateDirectory(outputDir);
        }

        // Build command arguments
        var args = new StringBuilder();
        args.Append($"\"{_converterScriptPath}\" ");
        args.Append($"--model_dir \"{modelDir}\" ");
        args.Append($"--output \"{outputPath}\" ");
        args.Append($"--model_filename \"{modelFilename}\" ");
        args.Append($"--params_filename \"{paramsFilename}\" ");
        args.Append($"--opset_version {opsetVersion}");

        if (!enableValidation)
        {
            args.Append(" --no-validation");
        }

        // Execute conversion
        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonPath,
            Arguments = args.ToString(),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = startInfo };
        
        var output = new StringBuilder();
        var error = new StringBuilder();

        process.OutputDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Console.WriteLine(e.Data);
                output.AppendLine(e.Data);
            }
        };

        process.ErrorDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Console.Error.WriteLine(e.Data);
                error.AppendLine(e.Data);
            }
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"PaddlePaddle to ONNX conversion failed with exit code {process.ExitCode}. " +
                $"Error: {error}");
        }

        if (!File.Exists(outputPath))
        {
            throw new FileNotFoundException(
                $"Conversion appeared to succeed but output file not found: {outputPath}");
        }
    }

    /// <summary>
    /// Check if paddle2onnx is installed
    /// </summary>
    public bool IsPaddle2OnnxInstalled()
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = _pythonPath,
                Arguments = "-c \"import paddle2onnx\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            process?.WaitForExit();
            return process?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get installation instructions for paddle2onnx
    /// </summary>
    public static string GetInstallationInstructions()
    {
        return @"
PaddlePaddle to ONNX conversion requires paddle2onnx:

1. Install Python 3.7 or higher
2. Install paddle2onnx and dependencies:
   pip install paddle2onnx onnx onnxruntime

For more information, visit:
https://github.com/PaddlePaddle/Paddle2ONNX
";
    }

    /// <summary>
    /// Clear the conversion cache
    /// </summary>
    public void ClearCache()
    {
        if (Directory.Exists(_cacheDir))
        {
            Directory.Delete(_cacheDir, recursive: true);
            Directory.CreateDirectory(_cacheDir);
        }
    }

    /// <summary>
    /// Get the size of the conversion cache in bytes
    /// </summary>
    public long GetCacheSize()
    {
        if (!Directory.Exists(_cacheDir))
            return 0;

        var dirInfo = new DirectoryInfo(_cacheDir);
        return dirInfo.GetFiles("*.onnx", SearchOption.AllDirectories)
                     .Sum(file => file.Length);
    }
}
