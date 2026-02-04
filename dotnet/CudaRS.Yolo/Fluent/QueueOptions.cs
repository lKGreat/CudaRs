namespace CudaRS.Fluent;

/// <summary>
/// 队列/背压配置，占位向下传递给后端（当前 JSON 透传，后端可选择支持）。
/// </summary>
public sealed class QueueOptions
{
    public int Capacity { get; set; } = 32;
    public int TimeoutMs { get; set; } = -1; // -1 表示阻塞等待
    public bool Backpressure { get; set; } = true;
}
