using CudaRS.Fluent;

namespace CudaRS;

/// <summary>
/// Entry point for fluent pipeline creation. Lives in CudaRS.Yolo so consumers add the backend package to access AsTensorRt/AsCpu/etc.
/// </summary>
public sealed class CudaRsFluent
{
    private CudaRsFluent() { }

    public static CudaRsFluent Create() => new();

    public FluentPipelineBuilder Pipeline() => new();
}
