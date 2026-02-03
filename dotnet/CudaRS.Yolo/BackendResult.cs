using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

public sealed class BackendResult
{
    public IReadOnlyList<TensorOutput> Outputs { get; init; } = Array.Empty<TensorOutput>();
}
