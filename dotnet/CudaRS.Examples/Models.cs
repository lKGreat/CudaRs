namespace CudaRS.Examples;

/// <summary>
/// 图片输入数据
/// </summary>
record ImageInput(string Path, byte[] Bytes);

/// <summary>
/// 基准测试结果
/// </summary>
record BenchmarkResult
{
    public required string Name { get; init; }
    public required int Iterations { get; init; }
    public required List<double> Times { get; init; }
    public int SuccessCount { get; init; }
    public int FailureCount { get; init; }
    
    public double Avg => Times.Count > 0 ? Times.Average() : 0;
    public double Median => BenchmarkHelpers.ComputeMedian(Times);
    public double SteadyAvg => Times.Count > 1 ? Times.Skip(1).Average() : Avg;
    public double FirstMs => Times.Count > 0 ? Times[0] : 0;
    public double TotalMs => Times.Sum();
}
