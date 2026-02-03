using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

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
