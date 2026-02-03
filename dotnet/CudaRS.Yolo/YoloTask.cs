namespace CudaRS.Yolo;

/// <summary>
/// YOLO task type.
/// </summary>
public enum YoloTask
{
    Detect = 0,
    Segment = 1,
    Pose = 2,
    Classify = 3,
    Obb = 4,
}
