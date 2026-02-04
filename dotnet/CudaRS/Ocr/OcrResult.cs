using System.Collections.Generic;

namespace CudaRS.Ocr;

public sealed class OcrResult
{
    public IReadOnlyList<OcrLine> Lines { get; init; } = new List<OcrLine>();
    public string? StructJson { get; init; }
}
