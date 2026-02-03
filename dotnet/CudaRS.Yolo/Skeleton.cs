namespace CudaRS.Yolo;

/// <summary>
/// Skeleton connection definitions for drawing.
/// </summary>
public static class Skeleton
{
    public static readonly (KeypointType From, KeypointType To)[] Connections =
    {
        (KeypointType.Nose, KeypointType.LeftEye),
        (KeypointType.Nose, KeypointType.RightEye),
        (KeypointType.LeftEye, KeypointType.LeftEar),
        (KeypointType.RightEye, KeypointType.RightEar),
        (KeypointType.LeftShoulder, KeypointType.RightShoulder),
        (KeypointType.LeftShoulder, KeypointType.LeftElbow),
        (KeypointType.RightShoulder, KeypointType.RightElbow),
        (KeypointType.LeftElbow, KeypointType.LeftWrist),
        (KeypointType.RightElbow, KeypointType.RightWrist),
        (KeypointType.LeftShoulder, KeypointType.LeftHip),
        (KeypointType.RightShoulder, KeypointType.RightHip),
        (KeypointType.LeftHip, KeypointType.RightHip),
        (KeypointType.LeftHip, KeypointType.LeftKnee),
        (KeypointType.RightHip, KeypointType.RightKnee),
        (KeypointType.LeftKnee, KeypointType.LeftAnkle),
        (KeypointType.RightKnee, KeypointType.RightAnkle),
    };

    public static readonly (byte R, byte G, byte B)[] LimbColors =
    {
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    };
}
