using System;
using System.Runtime.InteropServices;
using CudaRS.Native;

namespace CudaRS.Core;

/// <summary>
/// SafeHandle wrapper for cuBLAS handles.
/// </summary>
public sealed class CublasHandle : SafeHandle
{
    public CublasHandle() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.CublasCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.CublasDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Set the stream for this handle.
    /// </summary>
    public void SetStream(CudaStream stream)
    {
        CudaCheck.ThrowIfError(CudaRsNative.CublasSetStream((ulong)handle.ToInt64(), stream.Handle));
    }

    /// <summary>
    /// Get the cuBLAS version.
    /// </summary>
    public int Version
    {
        get
        {
            CudaCheck.ThrowIfError(CudaRsNative.CublasGetVersion((ulong)handle.ToInt64(), out int version));
            return version;
        }
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for cuFFT 1D plans.
/// </summary>
public sealed class CufftPlan1dC2C : SafeHandle
{
    public int Nx { get; }

    public CufftPlan1dC2C(int nx) : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.FftPlan1dC2C(out ulong handle, nx));
        SetHandle(new IntPtr((long)handle));
        Nx = nx;
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.FftPlan1dDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for cuFFT 2D plans.
/// </summary>
public sealed class CufftPlan2dC2C : SafeHandle
{
    public int Nx { get; }
    public int Ny { get; }

    public CufftPlan2dC2C(int nx, int ny) : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.FftPlan2dC2C(out ulong handle, nx, ny));
        SetHandle(new IntPtr((long)handle));
        Nx = nx;
        Ny = ny;
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.FftPlan2dDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// Random number generator types.
/// </summary>
public enum RngType
{
    PseudoXorwow = CudaRsNative.RngPseudoXorwow,
    PseudoMrg32k3a = CudaRsNative.RngPseudoMrg32k3a,
    PseudoPhilox = CudaRsNative.RngPseudoPhilox,
    QuasiSobol32 = CudaRsNative.RngQuasiSobol32,
}

/// <summary>
/// SafeHandle wrapper for cuRAND generators.
/// </summary>
public sealed class CurandGenerator : SafeHandle
{
    public CurandGenerator(RngType rngType = RngType.PseudoXorwow) : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.RandCreate(out ulong handle, (int)rngType));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.RandDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Set the seed.
    /// </summary>
    public void SetSeed(ulong seed)
    {
        CudaCheck.ThrowIfError(CudaRsNative.RandSetSeed((ulong)handle.ToInt64(), seed));
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for cuSPARSE handles.
/// </summary>
public sealed class CusparseHandle : SafeHandle
{
    public CusparseHandle() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.SparseCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.SparseDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Get the cuSPARSE version.
    /// </summary>
    public int Version
    {
        get
        {
            CudaCheck.ThrowIfError(CudaRsNative.SparseGetVersion((ulong)handle.ToInt64(), out int version));
            return version;
        }
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for cuSOLVER dense handles.
/// </summary>
public sealed class CusolverDnHandle : SafeHandle
{
    public CusolverDnHandle() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.SolverDnCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.SolverDnDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for cuDNN handles.
/// </summary>
public sealed class CudnnHandle : SafeHandle
{
    public CudnnHandle() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.CudnnCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.CudnnDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Get the cuDNN version.
    /// </summary>
    public static nuint Version => CudaRsNative.CudnnGetVersion();

    public ulong Handle => (ulong)handle.ToInt64();
}
