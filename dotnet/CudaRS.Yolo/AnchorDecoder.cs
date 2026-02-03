using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

/// <summary>
/// Anchor decoder for anchor-based YOLO models (V3/V4/V5/V6/V7).
/// </summary>
public static class AnchorDecoder
{
    /// <summary>
    /// Decodes anchor-based predictions to absolute coordinates.
    /// </summary>
    public static void DecodeAnchors(
        Span<float> output,
        int gridW,
        int gridH,
        int stride,
        float[] anchors,
        int numClasses,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        float confThreshold,
        bool hasObjectness)
    {
        var numAnchors = anchors.Length / 2;
        var channelsPerAnchor = (hasObjectness ? 5 : 4) + numClasses;

        for (int ay = 0; ay < gridH; ay++)
        {
            for (int ax = 0; ax < gridW; ax++)
            {
                for (int a = 0; a < numAnchors; a++)
                {
                    var offset = (ay * gridW * numAnchors + ax * numAnchors + a) * channelsPerAnchor;

                    float objectness = hasObjectness
                        ? Sigmoid(output[offset + 4])
                        : 1.0f;

                    if (objectness < confThreshold)
                        continue;

                    var classStart = hasObjectness ? 5 : 4;
                    var bestClass = 0;
                    var bestProb = 0f;

                    for (int c = 0; c < numClasses; c++)
                    {
                        var prob = Sigmoid(output[offset + classStart + c]);
                        if (prob > bestProb)
                        {
                            bestProb = prob;
                            bestClass = c;
                        }
                    }

                    var confidence = objectness * bestProb;
                    if (confidence < confThreshold)
                        continue;

                    var anchorW = anchors[a * 2];
                    var anchorH = anchors[a * 2 + 1];

                    var tx = Sigmoid(output[offset]);
                    var ty = Sigmoid(output[offset + 1]);
                    var tw = output[offset + 2];
                    var th = output[offset + 3];

                    var cx = (ax + tx) * stride;
                    var cy = (ay + ty) * stride;
                    var w = anchorW * MathF.Exp(tw);
                    var h = anchorH * MathF.Exp(th);

                    boxes.Add(BoundingBox.FromCenterWH(cx, cy, w, h));
                    scores.Add(confidence);
                    classIds.Add(bestClass);
                }
            }
        }
    }

    /// <summary>
    /// Decodes V10-style output (already post-NMS with XYXY format).
    /// </summary>
    public static void DecodeV10(
        Span<float> output,
        int numDetections,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        float confThreshold)
    {
        for (int i = 0; i < numDetections; i++)
        {
            var offset = i * 6;

            var x1 = output[offset];
            var y1 = output[offset + 1];
            var x2 = output[offset + 2];
            var y2 = output[offset + 3];
            var score = output[offset + 4];
            var classId = (int)output[offset + 5];

            if (score < confThreshold)
                continue;

            var box = new BoundingBox(x1, y1, x2 - x1, y2 - y1);
            boxes.Add(box);
            scores.Add(score);
            classIds.Add(classId);
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
