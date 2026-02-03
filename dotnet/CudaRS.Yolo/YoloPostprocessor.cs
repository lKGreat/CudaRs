using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

public static class YoloPostprocessor
{
    public static ModelInferenceResult Decode(
        string modelId,
        YoloConfig config,
        BackendResult backendResult,
        YoloPreprocessResult preprocess,
        string channelId,
        long frameIndex)
    {
        if (backendResult.Outputs.Count == 0)
            return new ModelInferenceResult { ModelId = modelId, ChannelId = channelId, FrameIndex = frameIndex, Success = false, ErrorMessage = "No outputs" };

        var main = backendResult.Outputs[0];
        var outputs = main.Data;
        var shape = main.Shape;

        var (channels, count, transposed) = NormalizeShape(shape);
        var classCount = GetClassCount(config, channels, transposed);
        var extra = GetExtraCount(config);

        var boxes = new List<BoundingBox>();
        var scores = new List<float>();
        var classIds = new List<int>();
        var extras = new List<float[]>();

        for (int i = 0; i < count; i++)
        {
            float x = GetValue(outputs, channels, count, i, 0, transposed);
            float y = GetValue(outputs, channels, count, i, 1, transposed);
            float w = GetValue(outputs, channels, count, i, 2, transposed);
            float h = GetValue(outputs, channels, count, i, 3, transposed);

            float obj = 1f;
            int classStart = 4;
            if (!config.AnchorFree)
            {
                obj = GetValue(outputs, channels, count, i, 4, transposed);
                classStart = 5;
            }

            var bestClass = 0;
            var bestScore = 0f;
            for (int c = 0; c < classCount; c++)
            {
                var score = GetValue(outputs, channels, count, i, classStart + c, transposed);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            var conf = bestScore * obj;
            if (conf < config.ConfidenceThreshold)
                continue;

            var box = BoundingBox.FromCenterWH(x, y, w, h);
            box = ScaleBox(box, preprocess, config);

            boxes.Add(box);
            scores.Add(conf);
            classIds.Add(bestClass);

            if (extra > 0)
            {
                var extraValues = new float[extra];
                for (int e = 0; e < extra; e++)
                {
                    extraValues[e] = GetValue(outputs, channels, count, i, classStart + classCount + e, transposed);
                }
                extras.Add(extraValues);
            }
            else
            {
                extras.Add(Array.Empty<float>());
            }
        }

        var selected = Nms.Apply(boxes, scores, config.IouThreshold, config.MaxDetections, config.ClassAgnosticNms, classIds);
        var nmsSummary = new NmsSummary
        {
            IouThreshold = config.IouThreshold,
            MaxDetections = config.MaxDetections,
            ClassAgnostic = config.ClassAgnosticNms,
            PreNmsCount = boxes.Count,
            PostNmsCount = selected.Count,
        };

        switch (config.Task)
        {
            case YoloTask.Segment:
                return BuildSegmentResult(modelId, config, preprocess, boxes, scores, classIds, extras, selected, backendResult, nmsSummary, channelId, frameIndex);
            case YoloTask.Pose:
                return BuildPoseResult(modelId, config, preprocess, boxes, scores, classIds, extras, selected, nmsSummary, channelId, frameIndex);
            case YoloTask.Obb:
                return BuildObbResult(modelId, config, preprocess, boxes, scores, classIds, extras, selected, nmsSummary, channelId, frameIndex);
            case YoloTask.Classify:
                return BuildClassifyResult(modelId, config, backendResult, channelId, frameIndex);
            default:
                return BuildDetectResult(modelId, config, preprocess, boxes, scores, classIds, selected, nmsSummary, channelId, frameIndex);
        }
    }

    private static (int channels, int count, bool transposed) NormalizeShape(int[] shape)
    {
        if (shape.Length == 3)
        {
            var c = shape[1];
            var n = shape[2];
            var transposed = false;
            if (shape[1] < shape[2])
            {
                c = shape[1];
                n = shape[2];
                transposed = true;
            }
            else
            {
                c = shape[2];
                n = shape[1];
            }
            return (c, n, transposed);
        }

        if (shape.Length == 2)
            return (shape[1], shape[0], true);

        return (shape.Last(), shape.First(), false);
    }

    private static int GetClassCount(YoloConfig config, int channels, bool transposed)
    {
        if (config.ClassNames.Length > 0)
            return config.ClassNames.Length;

        var baseCount = config.AnchorFree ? 4 : 5;
        var extra = GetExtraCount(config);
        var remaining = channels - baseCount - extra;
        return Math.Max(1, remaining);
    }

    private static int GetExtraCount(YoloConfig config)
        => config.Task switch
        {
            YoloTask.Segment => config.MaskProtoChannels,
            YoloTask.Pose => config.KeypointCount * 3,
            YoloTask.Obb => 1,
            _ => 0,
        };

    private static float GetValue(float[] data, int channels, int count, int index, int channel, bool transposed)
        => transposed ? data[channel * count + index] : data[index * channels + channel];

    private static BoundingBox ScaleBox(BoundingBox box, YoloPreprocessResult preprocess, YoloConfig config)
    {
        var x = (box.X - preprocess.PadX) / preprocess.Scale;
        var y = (box.Y - preprocess.PadY) / preprocess.Scale;
        var w = box.Width / preprocess.Scale;
        var h = box.Height / preprocess.Scale;

        return new BoundingBox(x, y, w, h).Clamp(preprocess.OriginalWidth, preprocess.OriginalHeight);
    }

    private static ModelInferenceResult BuildDetectResult(
        string modelId,
        YoloConfig config,
        YoloPreprocessResult preprocess,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        List<int> selected,
        NmsSummary nmsSummary,
        string channelId,
        long frameIndex)
    {
        var detections = selected.Select(i => new Detection
        {
            ClassId = classIds[i],
            ClassName = GetClassName(config, classIds[i]),
            Confidence = scores[i],
            Box = boxes[i],
            SourceWidth = preprocess.OriginalWidth,
            SourceHeight = preprocess.OriginalHeight,
        }).ToList();

        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Task = YoloTask.Detect,
            Version = config.Version,
            Success = true,
            Detections = detections,
            Nms = nmsSummary,
        };
    }

    private static ModelInferenceResult BuildSegmentResult(
        string modelId,
        YoloConfig config,
        YoloPreprocessResult preprocess,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        List<float[]> extras,
        List<int> selected,
        BackendResult backendResult,
        NmsSummary nmsSummary,
        string channelId,
        long frameIndex)
    {
        var proto = backendResult.Outputs.Count > 1 ? backendResult.Outputs[1] : null;
        var protoData = proto?.Data ?? Array.Empty<float>();
        var protoShape = proto?.Shape ?? new[] { 1, config.MaskProtoChannels, config.MaskProtoHeight, config.MaskProtoWidth };

        var segmentations = new List<SegmentationDetection>();
        foreach (var i in selected)
        {
            var mask = BuildMask(protoData, protoShape, extras[i], config);
            segmentations.Add(new SegmentationDetection
            {
                ClassId = classIds[i],
                ClassName = GetClassName(config, classIds[i]),
                Confidence = scores[i],
                Box = boxes[i],
                Mask = mask,
                SourceWidth = preprocess.OriginalWidth,
                SourceHeight = preprocess.OriginalHeight,
            });
        }

        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Task = YoloTask.Segment,
            Version = config.Version,
            Success = true,
            Segmentations = segmentations,
            Nms = nmsSummary,
        };
    }

    private static SegmentationMask BuildMask(float[] proto, int[] shape, float[] coeffs, YoloConfig config)
    {
        if (proto.Length == 0 || coeffs.Length == 0)
            return new SegmentationMask { Width = config.MaskProtoWidth, Height = config.MaskProtoHeight };

        var c = shape[1];
        var h = shape[2];
        var w = shape[3];
        var mask = new float[w * h];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0f;
                var idx = y * w + x;
                for (int k = 0; k < c; k++)
                {
                    var protoIdx = k * h * w + idx;
                    sum += proto[protoIdx] * coeffs[k];
                }
                mask[idx] = Sigmoid(sum);
            }
        }

        return new SegmentationMask { Width = w, Height = h, Data = mask };
    }

    private static ModelInferenceResult BuildPoseResult(
        string modelId,
        YoloConfig config,
        YoloPreprocessResult preprocess,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        List<float[]> extras,
        List<int> selected,
        NmsSummary nmsSummary,
        string channelId,
        long frameIndex)
    {
        var poses = new List<PoseDetection>();
        foreach (var i in selected)
        {
            var kp = extras[i];
            var keypoints = new List<Keypoint>();
            for (int k = 0; k < config.KeypointCount; k++)
            {
                var x = kp[k * 3 + 0];
                var y = kp[k * 3 + 1];
                var c = kp[k * 3 + 2];

                x = (x - preprocess.PadX) / preprocess.Scale;
                y = (y - preprocess.PadY) / preprocess.Scale;
                keypoints.Add(new Keypoint((KeypointType)k, x, y, c));
            }

            poses.Add(new PoseDetection
            {
                ClassId = classIds[i],
                ClassName = GetClassName(config, classIds[i]),
                Confidence = scores[i],
                Box = boxes[i],
                Pose = new Pose(keypoints),
                SourceWidth = preprocess.OriginalWidth,
                SourceHeight = preprocess.OriginalHeight,
            });
        }

        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Task = YoloTask.Pose,
            Version = config.Version,
            Success = true,
            Poses = poses,
            Nms = nmsSummary,
        };
    }

    private static ModelInferenceResult BuildObbResult(
        string modelId,
        YoloConfig config,
        YoloPreprocessResult preprocess,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        List<float[]> extras,
        List<int> selected,
        NmsSummary nmsSummary,
        string channelId,
        long frameIndex)
    {
        var obb = new List<ObbDetection>();
        foreach (var i in selected)
        {
            var angle = extras[i].Length > 0 ? extras[i][0] : 0f;
            if (config.AngleInRadians)
                angle = angle * 180f / MathF.PI;

            var box = boxes[i];
            var rbox = new RotatedBox(box.CenterX, box.CenterY, box.Width, box.Height, angle);

            obb.Add(new ObbDetection
            {
                ClassId = classIds[i],
                ClassName = GetClassName(config, classIds[i]),
                Confidence = scores[i],
                RotatedBox = rbox,
                SourceWidth = preprocess.OriginalWidth,
                SourceHeight = preprocess.OriginalHeight,
            });
        }

        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Task = YoloTask.Obb,
            Version = config.Version,
            Success = true,
            ObbDetections = obb,
            Nms = nmsSummary,
        };
    }

    private static ModelInferenceResult BuildClassifyResult(string modelId, YoloConfig config, BackendResult backendResult, string channelId, long frameIndex)
    {
        if (backendResult.Outputs.Count == 0)
            return new ModelInferenceResult { ModelId = modelId, ChannelId = channelId, FrameIndex = frameIndex, Task = YoloTask.Classify, Success = false };

        var data = backendResult.Outputs[0].Data;
        var classCount = data.Length;
        var classes = new List<Classification>(classCount);

        for (int i = 0; i < classCount; i++)
        {
            classes.Add(new Classification
            {
                ClassId = i,
                ClassName = GetClassName(config, i),
                Confidence = data[i],
            });
        }

        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Task = YoloTask.Classify,
            Version = config.Version,
            Success = true,
            Classifications = classes,
        };
    }

    private static string GetClassName(YoloConfig config, int id)
    {
        if (id >= 0 && id < config.ClassNames.Length)
            return config.ClassNames[id];
        return $"class_{id}";
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}
