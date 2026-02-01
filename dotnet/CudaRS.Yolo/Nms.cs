using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

public static class Nms
{
    public static List<int> Apply(IReadOnlyList<BoundingBox> boxes, IReadOnlyList<float> scores, float iouThreshold, int maxDetections, bool classAgnostic, IReadOnlyList<int>? classIds = null)
    {
        if (boxes.Count != scores.Count)
            throw new ArgumentException("Boxes and scores count must match.");

        var indices = Enumerable.Range(0, boxes.Count).OrderByDescending(i => scores[i]).ToList();
        var selected = new List<int>();

        while (indices.Count > 0 && selected.Count < maxDetections)
        {
            var current = indices[0];
            indices.RemoveAt(0);
            selected.Add(current);

            for (int i = indices.Count - 1; i >= 0; i--)
            {
                var idx = indices[i];
                if (!classAgnostic && classIds != null && classIds[current] != classIds[idx])
                    continue;

                var iou = boxes[current].IoU(boxes[idx]);
                if (iou >= iouThreshold)
                    indices.RemoveAt(i);
            }
        }

        return selected;
    }
}
