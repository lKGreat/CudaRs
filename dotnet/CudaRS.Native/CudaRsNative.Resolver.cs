using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

public static unsafe partial class CudaRsNative
{
    static CudaRsNative()
    {
        if (string.Equals(Environment.GetEnvironmentVariable("CUDARS_DIAG"), "1", StringComparison.Ordinal))
            Console.Error.WriteLine("[CudaRS] Initializing DllImportResolver for cudars_ffi");

        NativeLibrary.SetDllImportResolver(typeof(CudaRsNative).Assembly, ResolveCudaRs);
    }

    private static IntPtr ResolveCudaRs(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
            return IntPtr.Zero;

        if (string.Equals(Environment.GetEnvironmentVariable("CUDARS_DIAG"), "1", StringComparison.Ordinal))
            Console.Error.WriteLine($"[CudaRS] Resolving native library: {libraryName}");

        IntPtr nativeHandle;

        // Allow overriding via env vars.
        // - CUDARS_FFI_PATH: full path to cudars_ffi.dll
        // - CUDARS_FFI_DIR: directory containing cudars_ffi.dll
        var explicitPath = Environment.GetEnvironmentVariable("CUDARS_FFI_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath) && NativeLibrary.TryLoad(explicitPath, out nativeHandle))
            return nativeHandle;

        var explicitDir = Environment.GetEnvironmentVariable("CUDARS_FFI_DIR");
        if (!string.IsNullOrWhiteSpace(explicitDir))
        {
            var path = Path.Combine(explicitDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        // Common probe locations.
        var baseDir = AppContext.BaseDirectory;
        if (!string.IsNullOrWhiteSpace(baseDir))
        {
            var path = Path.Combine(baseDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;

            path = Path.Combine(baseDir, "runtimes", "win-x64", "native", "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        var assemblyDir = Path.GetDirectoryName(assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDir))
        {
            var path = Path.Combine(assemblyDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        // Fall back to default .NET resolution.
        if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out var handle2))
            return handle2;

        return IntPtr.Zero;
    }
}
