using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

// ============================================================
// Detection Result (Detect Task)
// ============================================================

/// <summary>
/// Single object detection result.
/// </summary>
public sealed class Detection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {Box}";
}

// ============================================================
// Segmentation Result (Segment Task)
// ============================================================

/// <summary>
/// Instance segmentation mask.
/// </summary>
public sealed class SegmentationMask
{
    public int Width { get; init; }
    public int Height { get; init; }
    public float[] Data { get; init; } = Array.Empty<float>();

    public bool[] ToBinary(float threshold = 0.5f)
        => Data.Select(v => v >= threshold).ToArray();

    public int GetArea(float threshold = 0.5f)
        => Data.Count(v => v >= threshold);

    public SegmentationMask ScaleTo(int targetWidth, int targetHeight)
    {
        if (Width == targetWidth && Height == targetHeight)
            return this;

        var scaled = new float[targetWidth * targetHeight];
        var scaleX = (float)Width / targetWidth;
        var scaleY = (float)Height / targetHeight;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                var srcX = (int)(x * scaleX);
                var srcY = (int)(y * scaleY);
                srcX = Math.Clamp(srcX, 0, Width - 1);
                srcY = Math.Clamp(srcY, 0, Height - 1);
                scaled[y * targetWidth + x] = Data[srcY * Width + srcX];
            }
        }

        return new SegmentationMask { Width = targetWidth, Height = targetHeight, Data = scaled };
    }

    public IReadOnlyList<Point2D> GetContour(float threshold = 0.5f)
    {
        var binary = ToBinary(threshold);
        var contour = new List<Point2D>();

        for (int y = 1; y < Height - 1; y++)
        {
            for (int x = 1; x < Width - 1; x++)
            {
                if (!binary[y * Width + x]) continue;

                var isEdge =
                    !binary[(y - 1) * Width + x] ||
                    !binary[(y + 1) * Width + x] ||
                    !binary[y * Width + (x - 1)] ||
                    !binary[y * Width + (x + 1)];

                if (isEdge)
                    contour.Add(new Point2D(x, y));
            }
        }

        return contour;
    }
}

/// <summary>
/// Segmentation detection result.
/// </summary>
public sealed class SegmentationDetection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public SegmentationMask Mask { get; init; } = null!;
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {Box} [mask {Mask.Width}x{Mask.Height}]";
}

// ============================================================
// Pose Estimation Result (Pose Task)
// ============================================================

/// <summary>
/// COCO keypoint types (17 points).
/// </summary>
public enum KeypointType
{
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

/// <summary>
/// Single keypoint with position and confidence.
/// </summary>
public readonly struct Keypoint
{
    public KeypointType Type { get; }
    public float X { get; }
    public float Y { get; }
    public float Confidence { get; }

    public bool IsVisible => Confidence > 0;
    public Point2D Position => new(X, Y);

    public Keypoint(KeypointType type, float x, float y, float confidence)
    {
        Type = type; X = x; Y = y; Confidence = confidence;
    }

    public Keypoint Scale(float sx, float sy)
        => new(Type, X * sx, Y * sy, Confidence);

    public override string ToString()
        => $"{Type}: ({X:F1},{Y:F1}) conf={Confidence:F2}";
}

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

/// <summary>
/// Human pose (17 keypoints).
/// </summary>
public sealed class Pose
{
    private readonly Dictionary<KeypointType, Keypoint> _keypoints;

    public IReadOnlyList<Keypoint> Keypoints { get; }

    public Pose(IEnumerable<Keypoint> keypoints)
    {
        var list = keypoints.ToList();
        Keypoints = list;
        _keypoints = list.ToDictionary(k => k.Type);
    }

    public Keypoint? GetKeypoint(KeypointType type)
        => _keypoints.TryGetValue(type, out var kp) ? kp : null;

    public int VisibleCount => Keypoints.Count(k => k.IsVisible);

    public float GetAverageConfidence()
    {
        var visible = Keypoints.Where(k => k.IsVisible).ToList();
        return visible.Count > 0 ? visible.Average(k => k.Confidence) : 0;
    }

    public IEnumerable<(Point2D Start, Point2D End, float Confidence)> GetSkeletonLines(float minConfidence = 0.3f)
    {
        foreach (var (from, to) in Skeleton.Connections)
        {
            var kpFrom = GetKeypoint(from);
            var kpTo = GetKeypoint(to);

            if (kpFrom is { IsVisible: true } && kpTo is { IsVisible: true })
            {
                var conf = Math.Min(kpFrom.Value.Confidence, kpTo.Value.Confidence);
                if (conf >= minConfidence)
                    yield return (kpFrom.Value.Position, kpTo.Value.Position, conf);
            }
        }
    }

    public Pose ScaleTo(float scaleX, float scaleY)
        => new(Keypoints.Select(k => k.Scale(scaleX, scaleY)));
}

/// <summary>
/// Pose detection result.
/// </summary>
public sealed class PoseDetection
{
    public int ClassId { get; init; } = 0;
    public string ClassName { get; init; } = "person";
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public Pose Pose { get; init; } = null!;
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}: {Confidence:P1} @ {Box} [{Pose.VisibleCount}/17 keypoints]";
}

// ============================================================
// OBB Detection Result (Obb Task)
// ============================================================

/// <summary>
/// Oriented bounding box detection result.
/// </summary>
public sealed class ObbDetection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public RotatedBox RotatedBox { get; init; }
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public BoundingBox AxisAlignedBox => RotatedBox.GetAxisAlignedBox();

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {RotatedBox}";
}

// ============================================================
// Classification Result (Classify Task)
// ============================================================

/// <summary>
/// Classification result.
/// </summary>
public sealed class Classification
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1}";
}
