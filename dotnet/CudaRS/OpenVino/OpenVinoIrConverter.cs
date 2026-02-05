using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace CudaRS.OpenVino;

/// <summary>
/// Converts ONNX models to OpenVINO IR using Python ovc.
/// </summary>
public sealed class OpenVinoIrConverter
{
    private readonly string _pythonPath;
    private readonly string _converterScriptPath;
    private readonly string _cacheDir;

    public OpenVinoIrConverter(string? pythonPath = null, string? cacheDir = null)
    {
        _pythonPath = pythonPath ?? "python";
        _cacheDir = cacheDir ?? Path.Combine(Path.GetTempPath(), "openvino_ir_cache");

        var scriptName = "onnx_to_openvino_ir.py";
        _converterScriptPath = FindScriptPath(scriptName);

        if (string.IsNullOrEmpty(_converterScriptPath))
        {
            throw new FileNotFoundException(
                $"Converter script '{scriptName}' not found. Ensure scripts folder exists near the repo root.");
        }

        if (!Directory.Exists(_cacheDir))
        {
            Directory.CreateDirectory(_cacheDir);
        }
    }

    public bool IsOpenVinoInstalled()
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = _pythonPath,
                Arguments = "-c \"import openvino.tools.ovc\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = startInfo };
            process.Start();
            process.WaitForExit();
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    public string ConvertOrUseCache(string onnxPath, bool forceReconvert = false, bool compressToFp16 = false)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxPath}");

        var info = new FileInfo(onnxPath);
        var modelName = Path.GetFileNameWithoutExtension(onnxPath);
        var cacheKey = $"{modelName}_{info.LastWriteTimeUtc:yyyyMMddHHmmss}_{(compressToFp16 ? "fp16" : "fp32")}";
        var outputDir = Path.Combine(_cacheDir, cacheKey);
        var xmlPath = Path.Combine(outputDir, $"{modelName}.xml");
        var binPath = Path.Combine(outputDir, $"{modelName}.bin");

        if (!forceReconvert && File.Exists(xmlPath) && File.Exists(binPath))
        {
            Console.WriteLine($"Using cached IR model: {xmlPath}");
            return xmlPath;
        }

        Convert(onnxPath, outputDir, modelName, compressToFp16);
        return xmlPath;
    }

    public string Convert(string onnxPath, string outputDir, string? modelName = null, bool compressToFp16 = false)
    {
        if (!File.Exists(onnxPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxPath}");

        Directory.CreateDirectory(outputDir);
        modelName ??= Path.GetFileNameWithoutExtension(onnxPath);

        var args = new StringBuilder();
        args.Append($"\"{_converterScriptPath}\" ");
        args.Append($"--input \"{onnxPath}\" ");
        args.Append($"--output_dir \"{outputDir}\" ");
        args.Append($"--model_name \"{modelName}\"");
        if (compressToFp16)
            args.Append(" --compress_to_fp16");

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

        process.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Console.WriteLine(e.Data);
                output.AppendLine(e.Data);
            }
        };
        process.ErrorDataReceived += (_, e) =>
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
                $"ONNX to IR conversion failed (exit {process.ExitCode}). Error: {error}");
        }

        var xmlPath = Path.Combine(outputDir, $"{modelName}.xml");
        var binPath = Path.Combine(outputDir, $"{modelName}.bin");
        if (!File.Exists(xmlPath) || !File.Exists(binPath))
            throw new FileNotFoundException($"IR output files not found: {xmlPath} / {binPath}");

        return xmlPath;
    }

    public static string GetInstallationInstructions()
        => "OpenVINO Python package required:\n" +
           "1. Install OpenVINO:\n" +
           "   pip install openvino\n" +
           "2. Verify:\n" +
           "   python -c \"import openvino.tools.ovc\"";

    private static string FindScriptPath(string scriptName)
    {
        var startDirs = new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() };
        foreach (var start in startDirs)
        {
            var dir = new DirectoryInfo(start);
            for (var i = 0; i < 7 && dir != null; i++)
            {
                var candidate = Path.Combine(dir.FullName, "scripts", scriptName);
                if (File.Exists(candidate))
                    return candidate;
                dir = dir.Parent;
            }
        }
        return string.Empty;
    }
}
