
use cudars_core::SdkErr;

use super::openvino_config_utils::{build_openvino_properties_json, parse_openvino_device};
use super::openvino_ocr_model_config::OpenVinoOcrModelConfig;
use super::openvino_output::OpenVinoOutput;
use super::openvino_pipeline_config::OpenVinoPipelineConfig;
use super::paddleocr_output::SdkOcrLine;
use super::sdk_error::{handle_cudars_result, set_last_error};
#[cfg(feature = "openvino")]
use super::yolo_preprocess_cpu::{decode_image_rgb, InputLayout};

#[cfg(feature = "openvino")]
mod imp {
    use super::*;
    use crate::{
        cudars_ov_destroy, cudars_ov_free_tensors, cudars_ov_get_input_info, cudars_ov_load_v2,
        cudars_ov_run, CudaRsOvConfigV2, CudaRsOvModel, CudaRsOvTensor, CudaRsOvTensorInfo, CudaRsResult,
    };
    use std::cmp::{max, min};
    use std::collections::VecDeque;
    use std::ffi::CString;
    use std::fs;
    use std::ptr;

    #[derive(Clone, Copy)]
    struct DetResizeMeta {
        orig_w: i32,
        orig_h: i32,
        resized_w: i32,
        resized_h: i32,
        pad_w: i32,
        pad_h: i32,
        scale: f32,
    }

    #[derive(Clone, Copy)]
    struct OcrBox {
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        score: f32,
    }

    pub struct OpenVinoOcrPipeline {
        det_model: CudaRsOvModel,
        rec_model: CudaRsOvModel,
        det_layout: InputLayout,
        rec_layout: InputLayout,
        det_channels: i32,
        dict: Vec<String>,
        config: OpenVinoOcrModelConfig,
        lines: Vec<SdkOcrLine>,
        text_buffer: Vec<u8>,
        struct_json: Option<String>,
    }

    unsafe impl Send for OpenVinoOcrPipeline {}

    impl OpenVinoOcrPipeline {
        pub fn new(model: &OpenVinoOcrModelConfig, pipeline: &OpenVinoPipelineConfig) -> Result<Self, SdkErr> {
            if model.det_model_path.is_empty() || model.rec_model_path.is_empty() {
                set_last_error("det_model_path and rec_model_path are required");
                return Err(SdkErr::InvalidArg);
            }
            if model.dict_path.is_empty() {
                set_last_error("dict_path is required");
                return Err(SdkErr::InvalidArg);
            }

            let dict = load_dict(&model.dict_path)?;

            let det_model = load_ov_model(&model.det_model_path, model, pipeline)?;
            let rec_model = load_ov_model(&model.rec_model_path, model, pipeline)?;

            let det_layout = query_input_layout(det_model, 3);
            let rec_layout = query_input_layout(rec_model, 3);

            let det_channels = query_input_channels(det_model).unwrap_or(3);
            let rec_channels = query_input_channels(rec_model).unwrap_or(3);
            if det_channels != 3 || rec_channels != 3 {
                set_last_error("OpenVINO OCR expects 3-channel input");
                return Err(SdkErr::Unsupported);
            }

            Ok(Self {
                det_model,
                rec_model,
                det_layout,
                rec_layout,
                det_channels,
                dict,
                config: model.clone(),
                lines: Vec::new(),
                text_buffer: Vec::new(),
                struct_json: None,
            })
        }

        pub fn run_image(&mut self, data: *const u8, len: usize) -> SdkErr {
            if data.is_null() || len == 0 {
                set_last_error("input data is null or empty");
                return SdkErr::InvalidArg;
            }

            let bytes = unsafe { std::slice::from_raw_parts(data, len) };
            let (rgb, width, height) = match decode_image_rgb(bytes) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let (det_input, det_meta) = match prepare_det_input(
                &rgb,
                width,
                height,
                self.config.det_resize_long,
                self.config.det_stride,
                self.det_layout,
            ) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let det_shape = input_shape(
                self.det_layout,
                1,
                self.det_channels,
                det_meta.pad_h,
                det_meta.pad_w,
            );

            let det_outputs = match run_openvino(self.det_model, &det_input, &det_shape) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let (prob_map, map_w, map_h) = match extract_prob_map(&det_outputs) {
                Ok(v) => v,
                Err(err) => return err,
            };

            let mut boxes = find_boxes(
                &prob_map,
                map_w,
                map_h,
                self.config.det_thresh,
                self.config.det_box_thresh,
                self.config.det_min_area,
                self.config.det_max_candidates,
            );

            if boxes.is_empty() {
                self.lines.clear();
                self.text_buffer.clear();
                self.struct_json = None;
                return SdkErr::Ok;
            }

            boxes = map_boxes(&boxes, map_w, map_h, &det_meta, self.config.det_box_padding);
            boxes.sort_by(|a, b| {
                let ay = a.y0;
                let by = b.y0;
                if ay == by {
                    a.x0.cmp(&b.x0)
                } else {
                    ay.cmp(&by)
                }
            });

            let rec_batch = max(1, self.config.rec_batch_size) as usize;
            let rec_h = max(1, self.config.rec_input_h);
            let rec_w = max(1, self.config.rec_input_w);

            let mut line_items: Vec<(OcrBox, String, f32)> = Vec::with_capacity(boxes.len());

            for chunk in boxes.chunks(rec_batch) {
                let mut batch_inputs: Vec<Vec<f32>> = Vec::with_capacity(chunk.len());
                for b in chunk {
                    let crop = crop_rgb(&rgb, width, height, b.x0, b.y0, b.x1, b.y1);
                    let input = match prepare_rec_input(&crop.0, crop.1, crop.2, rec_h, rec_w, self.rec_layout) {
                        Ok(v) => v,
                        Err(err) => return err,
                    };
                    batch_inputs.push(input);
                }

                let batch_size = batch_inputs.len();
                if batch_size == 0 {
                    continue;
                }

                let (batch_tensor, shape) = pack_batch(&batch_inputs, self.rec_layout, rec_h, rec_w, batch_size);
                let rec_outputs = match run_openvino(self.rec_model, &batch_tensor, &shape) {
                    Ok(v) => v,
                    Err(err) => return err,
                };

                let decoded = match decode_rec_outputs(&rec_outputs, batch_size, &self.dict) {
                    Ok(v) => v,
                    Err(err) => return err,
                };

                for (idx, (text, score)) in decoded.into_iter().enumerate() {
                    let b = chunk[idx];
                    if score < self.config.rec_score_thresh {
                        continue;
                    }
                    line_items.push((b, text, score));
                }
            }

            self.lines.clear();
            self.text_buffer.clear();
            self.struct_json = None;

            let mut text_offset: u32 = 0;
            for (b, text, score) in line_items {
                let bytes = text.as_bytes();
                let len = bytes.len().min(u32::MAX as usize) as u32;
                let points = [
                    b.x0 as f32,
                    b.y0 as f32,
                    b.x1 as f32,
                    b.y0 as f32,
                    b.x1 as f32,
                    b.y1 as f32,
                    b.x0 as f32,
                    b.y1 as f32,
                ];
                self.lines.push(SdkOcrLine {
                    points,
                    score: b.score,
                    cls_label: 0,
                    cls_score: score,
                    text_offset,
                    text_len: len,
                });
                if len > 0 {
                    self.text_buffer.extend_from_slice(&bytes[..len as usize]);
                    text_offset = text_offset.saturating_add(len);
                }
            }

            SdkErr::Ok
        }

        pub fn line_count(&self) -> Result<usize, SdkErr> {
            Ok(self.lines.len())
        }

        pub fn write_lines(&self, dst: *mut SdkOcrLine, cap: usize, out_written: *mut usize) -> SdkErr {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let total = self.lines.len();
            let to_copy = min(total, cap);
            unsafe {
                ptr::copy_nonoverlapping(self.lines.as_ptr(), dst, to_copy);
                *out_written = to_copy;
            }
            if cap < total {
                set_last_error("destination buffer too small");
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }

        pub fn text_bytes(&self) -> Result<usize, SdkErr> {
            Ok(self.text_buffer.len())
        }

        pub fn write_text(&self, dst: *mut i8, cap: usize, out_written: *mut usize) -> SdkErr {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let total = self.text_buffer.len();
            let to_copy = min(total, cap);
            unsafe {
                ptr::copy_nonoverlapping(self.text_buffer.as_ptr(), dst as *mut u8, to_copy);
                *out_written = to_copy;
            }
            if cap < total {
                set_last_error("destination buffer too small");
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }

        pub fn struct_json_bytes(&self) -> Result<usize, SdkErr> {
            Ok(self.struct_json.as_ref().map(|s| s.len()).unwrap_or(0))
        }

        pub fn write_struct_json(&self, dst: *mut i8, cap: usize, out_written: *mut usize) -> SdkErr {
            if dst.is_null() || out_written.is_null() {
                set_last_error("dst or out_written is null");
                return SdkErr::InvalidArg;
            }
            let json = match self.struct_json.as_ref() {
                Some(v) => v.as_bytes(),
                None => &[],
            };
            let to_copy = min(json.len(), cap);
            unsafe {
                ptr::copy_nonoverlapping(json.as_ptr(), dst as *mut u8, to_copy);
                *out_written = to_copy;
            }
            if cap < json.len() {
                set_last_error("destination buffer too small");
                return SdkErr::InvalidArg;
            }
            SdkErr::Ok
        }
    }

    impl Drop for OpenVinoOcrPipeline {
        fn drop(&mut self) {
            if self.det_model != 0 {
                let _ = cudars_ov_destroy(self.det_model);
            }
            if self.rec_model != 0 {
                let _ = cudars_ov_destroy(self.rec_model);
            }
        }
    }

    fn load_dict(path: &str) -> Result<Vec<String>, SdkErr> {
        let content = fs::read_to_string(path).map_err(|_| {
            set_last_error("failed to read dict file");
            SdkErr::InvalidArg
        })?;
        let mut dict = Vec::new();
        for line in content.lines() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            dict.push(line.to_string());
        }
        if dict.is_empty() {
            set_last_error("dict file is empty");
            return Err(SdkErr::InvalidArg);
        }
        Ok(dict)
    }

    fn load_ov_model(
        model_path: &str,
        model_cfg: &OpenVinoOcrModelConfig,
        pipeline: &OpenVinoPipelineConfig,
    ) -> Result<CudaRsOvModel, SdkErr> {
        let device = if !pipeline.openvino_device.trim().is_empty() {
            pipeline.openvino_device.as_str()
        } else if let Some(ref d) = model_cfg.device {
            d.as_str()
        } else {
            "cpu"
        };

        let device_spec = parse_openvino_device(device)?;
        let mut config = CudaRsOvConfigV2 {
            struct_size: std::mem::size_of::<CudaRsOvConfigV2>() as u32,
            device: device_spec.device,
            device_index: device_spec.device_index,
            device_name_ptr: ptr::null(),
            device_name_len: 0,
            num_streams: pipeline.openvino_num_streams,
            enable_profiling: if pipeline.openvino_enable_profiling { 1 } else { 0 },
            properties_json_ptr: ptr::null(),
            properties_json_len: 0,
        };

        let num_requests = if pipeline.openvino_num_requests > 0 {
            pipeline.openvino_num_requests
        } else {
            1
        };

        let perf_mode = if !pipeline.openvino_performance_mode.trim().is_empty() {
            pipeline.openvino_performance_mode.as_str()
        } else if num_requests > 1 || pipeline.openvino_num_streams > 1 {
            "throughput"
        } else {
            ""
        };

        let properties_json = build_openvino_properties_json(
            pipeline.openvino_config_json.as_str(),
            perf_mode,
            num_requests,
            pipeline.openvino_cache_dir.as_str(),
            pipeline.openvino_enable_mmap,
        )?;

        let mut props_cstr: Option<CString> = None;
        if let Some(json) = properties_json {
            let cstr = CString::new(json.as_str()).map_err(|_| SdkErr::InvalidArg)?;
            config.properties_json_ptr = cstr.as_ptr();
            config.properties_json_len = json.len();
            props_cstr = Some(cstr);
        }

        let mut device_name_cstr: Option<CString> = None;
        if let Some(name) = device_spec.device_name_override {
            let cstr = CString::new(name.as_str()).map_err(|_| SdkErr::InvalidArg)?;
            config.device_name_ptr = cstr.as_ptr();
            config.device_name_len = name.len();
            device_name_cstr = Some(cstr);
        }

        let model_path = CString::new(model_path).map_err(|_| SdkErr::InvalidArg)?;
        let mut handle: CudaRsOvModel = 0;
        let result = cudars_ov_load_v2(model_path.as_ptr(), &config, &mut handle);
        drop(device_name_cstr);
        drop(props_cstr);
        let err = handle_cudars_result(result, "openvino load");
        if err != SdkErr::Ok {
            return Err(err);
        }
        Ok(handle)
    }

    fn run_openvino(
        model: CudaRsOvModel,
        input: &[f32],
        shape: &[i64],
    ) -> Result<Vec<OpenVinoOutput>, SdkErr> {
        let mut out_tensors: *mut CudaRsOvTensor = ptr::null_mut();
        let mut out_count: u64 = 0;
        let result = cudars_ov_run(
            model,
            input.as_ptr(),
            input.len() as u64,
            shape.as_ptr(),
            shape.len() as u64,
            &mut out_tensors,
            &mut out_count,
        );
        let err = handle_cudars_result(result, "openvino run");
        if err != SdkErr::Ok {
            return Err(err);
        }

        let mut outputs = Vec::new();
        unsafe {
            let slice = std::slice::from_raw_parts(out_tensors, out_count as usize);
            outputs.reserve(slice.len());
            for t in slice {
                let data = std::slice::from_raw_parts(t.data, t.data_len as usize).to_vec();
                let shape = std::slice::from_raw_parts(t.shape, t.shape_len as usize).to_vec();
                outputs.push(OpenVinoOutput { shape, data });
            }
            let _ = cudars_ov_free_tensors(out_tensors, out_count as u64);
        }

        Ok(outputs)
    }

    fn query_input_layout(handle: CudaRsOvModel, channels: i32) -> InputLayout {
        let mut info = CudaRsOvTensorInfo {
            name_ptr: ptr::null_mut(),
            name_len: 0,
            shape: ptr::null_mut(),
            shape_len: 0,
            element_type: 0,
        };

        let result = cudars_ov_get_input_info(handle, 0, &mut info);
        if result != CudaRsResult::Success || info.shape.is_null() || info.shape_len == 0 {
            return InputLayout::Nchw;
        }

        unsafe {
            let shape_slice = std::slice::from_raw_parts(info.shape, info.shape_len as usize);
            infer_layout_from_shape(shape_slice, channels)
        }
    }

    fn infer_layout_from_shape(shape: &[i64], channels: i32) -> InputLayout {
        if shape.len() == 4 {
            if shape.get(1).copied().unwrap_or(0) == channels as i64 {
                return InputLayout::Nchw;
            }
            if shape.get(3).copied().unwrap_or(0) == channels as i64 {
                return InputLayout::Nhwc;
            }
        }
        InputLayout::Nchw
    }

    fn query_input_channels(handle: CudaRsOvModel) -> Option<i32> {
        let mut info = CudaRsOvTensorInfo {
            name_ptr: ptr::null_mut(),
            name_len: 0,
            shape: ptr::null_mut(),
            shape_len: 0,
            element_type: 0,
        };
        let result = cudars_ov_get_input_info(handle, 0, &mut info);
        if result != CudaRsResult::Success || info.shape.is_null() || info.shape_len == 0 {
            return None;
        }
        unsafe {
            let shape = std::slice::from_raw_parts(info.shape, info.shape_len as usize);
            if shape.len() == 4 {
                if shape[1] > 0 {
                    return Some(shape[1] as i32);
                }
                if shape[3] > 0 {
                    return Some(shape[3] as i32);
                }
            }
        }
        None
    }

    fn input_shape(layout: InputLayout, batch: i32, channels: i32, height: i32, width: i32) -> Vec<i64> {
        match layout {
            InputLayout::Nchw => vec![batch as i64, channels as i64, height as i64, width as i64],
            InputLayout::Nhwc => vec![batch as i64, height as i64, width as i64, channels as i64],
        }
    }

    fn prepare_det_input(
        rgb: &[u8],
        input_w: i32,
        input_h: i32,
        resize_long: i32,
        stride: i32,
        layout: InputLayout,
    ) -> Result<(Vec<f32>, DetResizeMeta), SdkErr> {
        let (resized, resized_w, resized_h, scale) = resize_long_side(rgb, input_w, input_h, resize_long)?;
        let pad_w = ((resized_w + stride - 1) / stride) * stride;
        let pad_h = ((resized_h + stride - 1) / stride) * stride;
        let padded = pad_rgb(&resized, resized_w, resized_h, pad_w, pad_h);
        let input = normalize_to_tensor(&padded, pad_w, pad_h, layout);
        let meta = DetResizeMeta {
            orig_w: input_w,
            orig_h: input_h,
            resized_w,
            resized_h,
            pad_w,
            pad_h,
            scale,
        };
        Ok((input, meta))
    }

    fn resize_long_side(rgb: &[u8], w: i32, h: i32, long_side: i32) -> Result<(Vec<u8>, i32, i32, f32), SdkErr> {
        if w <= 0 || h <= 0 {
            set_last_error("invalid image dimensions");
            return Err(SdkErr::InvalidArg);
        }
        let max_side = max(w, h) as f32;
        let scale = if long_side > 0 && max_side > 0.0 {
            long_side as f32 / max_side
        } else {
            1.0
        };
        let new_w = max(1, (w as f32 * scale).round() as i32);
        let new_h = max(1, (h as f32 * scale).round() as i32);
        let resized = resize_rgb_bilinear(rgb, w, h, new_w, new_h)?;
        Ok((resized, new_w, new_h, scale))
    }
    fn resize_rgb_bilinear(rgb: &[u8], w: i32, h: i32, out_w: i32, out_h: i32) -> Result<Vec<u8>, SdkErr> {
        if w <= 0 || h <= 0 || out_w <= 0 || out_h <= 0 {
            set_last_error("invalid resize dimensions");
            return Err(SdkErr::InvalidArg);
        }
        let mut output = vec![0u8; (out_w as usize) * (out_h as usize) * 3];
        let scale_x = w as f32 / out_w as f32;
        let scale_y = h as f32 / out_h as f32;
        for y in 0..out_h {
            let src_y = (y as f32 + 0.5) * scale_y - 0.5;
            let mut y0 = src_y.floor() as i32;
            let mut y1 = y0 + 1;
            let fy = src_y - y0 as f32;
            if y0 < 0 {
                y0 = 0;
            }
            if y1 >= h {
                y1 = h - 1;
            }
            for x in 0..out_w {
                let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                let mut x0 = src_x.floor() as i32;
                let mut x1 = x0 + 1;
                let fx = src_x - x0 as f32;
                if x0 < 0 {
                    x0 = 0;
                }
                if x1 >= w {
                    x1 = w - 1;
                }

                let idx00 = ((y0 * w + x0) as usize) * 3;
                let idx01 = ((y0 * w + x1) as usize) * 3;
                let idx10 = ((y1 * w + x0) as usize) * 3;
                let idx11 = ((y1 * w + x1) as usize) * 3;

                let w00 = (1.0 - fx) * (1.0 - fy);
                let w01 = fx * (1.0 - fy);
                let w10 = (1.0 - fx) * fy;
                let w11 = fx * fy;

                let out_idx = ((y * out_w + x) as usize) * 3;
                for c in 0..3 {
                    let v = rgb[idx00 + c] as f32 * w00
                        + rgb[idx01 + c] as f32 * w01
                        + rgb[idx10 + c] as f32 * w10
                        + rgb[idx11 + c] as f32 * w11;
                    output[out_idx + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(output)
    }

    fn pad_rgb(rgb: &[u8], w: i32, h: i32, pad_w: i32, pad_h: i32) -> Vec<u8> {
        let mut out = vec![0u8; (pad_w as usize) * (pad_h as usize) * 3];
        for y in 0..h {
            let src = (y as usize) * (w as usize) * 3;
            let dst = (y as usize) * (pad_w as usize) * 3;
            let len = (w as usize) * 3;
            out[dst..dst + len].copy_from_slice(&rgb[src..src + len]);
        }
        out
    }

    fn normalize_to_tensor(rgb: &[u8], w: i32, h: i32, layout: InputLayout) -> Vec<f32> {
        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];
        let hw = (w as usize) * (h as usize);
        let mut out = match layout {
            InputLayout::Nchw => vec![0f32; hw * 3],
            InputLayout::Nhwc => vec![0f32; hw * 3],
        };

        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) as usize) * 3;
                let r = rgb[idx] as f32 / 255.0;
                let g = rgb[idx + 1] as f32 / 255.0;
                let b = rgb[idx + 2] as f32 / 255.0;
                let r = (r - mean[0]) / std[0];
                let g = (g - mean[1]) / std[1];
                let b = (b - mean[2]) / std[2];
                match layout {
                    InputLayout::Nchw => {
                        let out_idx = (y as usize) * (w as usize) + (x as usize);
                        out[out_idx] = r;
                        out[out_idx + hw] = g;
                        out[out_idx + hw * 2] = b;
                    }
                    InputLayout::Nhwc => {
                        out[idx] = r;
                        out[idx + 1] = g;
                        out[idx + 2] = b;
                    }
                }
            }
        }
        out
    }

    fn extract_prob_map(outputs: &[OpenVinoOutput]) -> Result<(Vec<f32>, usize, usize), SdkErr> {
        if outputs.is_empty() {
            set_last_error("det model produced no outputs");
            return Err(SdkErr::Runtime);
        }
        let output = &outputs[0];
        let shape = output.shape.as_slice();
        if shape.len() < 2 {
            set_last_error("invalid det output shape");
            return Err(SdkErr::Runtime);
        }

        let (h, w, layout_nhwc) = if shape.len() == 4 {
            let n = shape[0] as usize;
            let c1 = shape[1] as usize;
            let h = shape[2] as usize;
            let w = shape[3] as usize;
            if n == 0 || h == 0 || w == 0 {
                set_last_error("invalid det output shape");
                return Err(SdkErr::Runtime);
            }
            if c1 == 1 {
                (h, w, false)
            } else if shape[3] == 1 {
                (shape[1] as usize, shape[2] as usize, true)
            } else {
                (h, w, false)
            }
        } else if shape.len() == 3 {
            let d0 = shape[0] as usize;
            let d1 = shape[1] as usize;
            let d2 = shape[2] as usize;
            if d2 == 1 {
                (d0, d1, true)
            } else {
                (d1, d2, false)
            }
        } else {
            set_last_error("unsupported det output shape");
            return Err(SdkErr::Runtime);
        };

        let mut map = vec![0f32; h * w];
        if layout_nhwc && shape.len() == 4 {
            let n = shape[0] as usize;
            let h2 = shape[1] as usize;
            let w2 = shape[2] as usize;
            let c = shape[3] as usize;
            if n != 1 || c == 0 {
                set_last_error("unsupported det output layout");
                return Err(SdkErr::Runtime);
            }
            for y in 0..h2 {
                for x in 0..w2 {
                    let idx = ((y * w2 + x) * c) as usize;
                    map[y * w2 + x] = output.data[idx];
                }
            }
            return Ok((map, w2, h2));
        }

        if shape.len() == 4 {
            let n = shape[0] as usize;
            let c = shape[1] as usize;
            let h2 = shape[2] as usize;
            let w2 = shape[3] as usize;
            if n != 1 || c == 0 {
                set_last_error("unsupported det output layout");
                return Err(SdkErr::Runtime);
            }
            for y in 0..h2 {
                for x in 0..w2 {
                    let idx = ((0 * c + 0) * h2 + y) * w2 + x;
                    map[y * w2 + x] = output.data[idx];
                }
            }
            return Ok((map, w2, h2));
        }

        set_last_error("unsupported det output layout");
        Err(SdkErr::Runtime)
    }

    fn find_boxes(
        prob: &[f32],
        w: usize,
        h: usize,
        thresh: f32,
        box_thresh: f32,
        min_area: i32,
        max_candidates: i32,
    ) -> Vec<OcrBox> {
        let mut boxes = Vec::new();
        let mut visited = vec![false; w * h];
        let mut queue = VecDeque::new();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if visited[idx] || prob[idx] <= thresh {
                    continue;
                }
                visited[idx] = true;
                queue.push_back((x as i32, y as i32));

                let mut minx = x as i32;
                let mut maxx = x as i32;
                let mut miny = y as i32;
                let mut maxy = y as i32;
                let mut sum = 0f32;
                let mut count = 0f32;

                while let Some((cx, cy)) = queue.pop_front() {
                    let cidx = (cy as usize) * w + (cx as usize);
                    let v = prob[cidx];
                    sum += v;
                    count += 1.0;
                    minx = min(minx, cx);
                    maxx = max(maxx, cx);
                    miny = min(miny, cy);
                    maxy = max(maxy, cy);

                    for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nidx = (ny as usize) * w + (nx as usize);
                        if visited[nidx] || prob[nidx] <= thresh {
                            continue;
                        }
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }

                let area = (maxx - minx + 1) * (maxy - miny + 1);
                if area < min_area {
                    continue;
                }
                let score = if count > 0.0 { sum / count } else { 0.0 };
                if score < box_thresh {
                    continue;
                }
                boxes.push(OcrBox {
                    x0: minx,
                    y0: miny,
                    x1: maxx,
                    y1: maxy,
                    score,
                });
                if max_candidates > 0 && boxes.len() >= max_candidates as usize {
                    return boxes;
                }
            }
        }

        boxes
    }

    fn map_boxes(
        boxes: &[OcrBox],
        map_w: usize,
        map_h: usize,
        meta: &DetResizeMeta,
        padding: i32,
    ) -> Vec<OcrBox> {
        let mut out = Vec::with_capacity(boxes.len());
        let scale_x = meta.pad_w as f32 / map_w as f32;
        let scale_y = meta.pad_h as f32 / map_h as f32;

        for b in boxes {
            let mut x0 = (b.x0 as f32) * scale_x;
            let mut x1 = (b.x1 as f32 + 1.0) * scale_x;
            let mut y0 = (b.y0 as f32) * scale_y;
            let mut y1 = (b.y1 as f32 + 1.0) * scale_y;

            x0 = x0.clamp(0.0, meta.resized_w as f32 - 1.0);
            x1 = x1.clamp(0.0, meta.resized_w as f32 - 1.0);
            y0 = y0.clamp(0.0, meta.resized_h as f32 - 1.0);
            y1 = y1.clamp(0.0, meta.resized_h as f32 - 1.0);

            x0 = x0 / meta.scale;
            x1 = x1 / meta.scale;
            y0 = y0 / meta.scale;
            y1 = y1 / meta.scale;

            let mut x0 = x0.floor() as i32 - padding;
            let mut y0 = y0.floor() as i32 - padding;
            let mut x1 = x1.ceil() as i32 + padding;
            let mut y1 = y1.ceil() as i32 + padding;

            x0 = x0.clamp(0, meta.orig_w - 1);
            y0 = y0.clamp(0, meta.orig_h - 1);
            x1 = x1.clamp(0, meta.orig_w - 1);
            y1 = y1.clamp(0, meta.orig_h - 1);

            if x1 <= x0 || y1 <= y0 {
                continue;
            }

            out.push(OcrBox { x0, y0, x1, y1, score: b.score });
        }
        out
    }

    fn crop_rgb(rgb: &[u8], w: i32, h: i32, x0: i32, y0: i32, x1: i32, y1: i32) -> (Vec<u8>, i32, i32) {
        let x0 = x0.clamp(0, w - 1);
        let y0 = y0.clamp(0, h - 1);
        let x1 = x1.clamp(0, w - 1);
        let y1 = y1.clamp(0, h - 1);
        let cw = max(1, x1 - x0);
        let ch = max(1, y1 - y0);

        let mut out = vec![0u8; (cw as usize) * (ch as usize) * 3];
        for y in 0..ch {
            let src_y = y0 + y;
            let src_idx = (src_y as usize) * (w as usize) * 3 + (x0 as usize) * 3;
            let dst_idx = (y as usize) * (cw as usize) * 3;
            let len = (cw as usize) * 3;
            out[dst_idx..dst_idx + len].copy_from_slice(&rgb[src_idx..src_idx + len]);
        }
        (out, cw, ch)
    }

    fn prepare_rec_input(
        rgb: &[u8],
        w: i32,
        h: i32,
        target_h: i32,
        target_w: i32,
        layout: InputLayout,
    ) -> Result<Vec<f32>, SdkErr> {
        let scale = target_h as f32 / h as f32;
        let new_w = max(1, min(target_w, (w as f32 * scale).round() as i32));
        let resized = resize_rgb_bilinear(rgb, w, h, new_w, target_h)?;
        let padded = pad_rgb(&resized, new_w, target_h, target_w, target_h);
        Ok(normalize_to_tensor(&padded, target_w, target_h, layout))
    }

    fn pack_batch(
        batch_inputs: &[Vec<f32>],
        layout: InputLayout,
        h: i32,
        w: i32,
        batch_size: usize,
    ) -> (Vec<f32>, Vec<i64>) {
        let single = (h as usize) * (w as usize) * 3;
        let mut tensor = vec![0f32; batch_size * single];
        for (i, input) in batch_inputs.iter().enumerate() {
            let dst = i * single;
            tensor[dst..dst + single].copy_from_slice(input);
        }
        let shape = match layout {
            InputLayout::Nchw => vec![batch_size as i64, 3, h as i64, w as i64],
            InputLayout::Nhwc => vec![batch_size as i64, h as i64, w as i64, 3],
        };
        (tensor, shape)
    }

    fn decode_rec_outputs(
        outputs: &[OpenVinoOutput],
        batch_size: usize,
        dict: &[String],
    ) -> Result<Vec<(String, f32)>, SdkErr> {
        if outputs.is_empty() {
            set_last_error("rec model produced no outputs");
            return Err(SdkErr::Runtime);
        }
        let output = &outputs[0];
        let shape = output.shape.as_slice();
        if shape.len() != 3 {
            set_last_error("unsupported rec output shape");
            return Err(SdkErr::Runtime);
        }

        let n = shape[0] as usize;
        let d1 = shape[1] as usize;
        let d2 = shape[2] as usize;
        if n == 0 || d1 == 0 || d2 == 0 {
            set_last_error("invalid rec output shape");
            return Err(SdkErr::Runtime);
        }

        let (seq_len, classes, layout_ntc) = if d2 > d1 {
            (d1, d2, true)
        } else {
            (d2, d1, false)
        };

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size.min(n) {
            let (text, score) = decode_ctc_single(&output.data, b, seq_len, classes, layout_ntc, dict);
            results.push((text, score));
        }
        Ok(results)
    }

    fn decode_ctc_single(
        data: &[f32],
        batch_idx: usize,
        seq_len: usize,
        classes: usize,
        layout_ntc: bool,
        dict: &[String],
    ) -> (String, f32) {
        let mut text = String::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut prev = -1i32;

        for t in 0..seq_len {
            let mut max_idx = 0usize;
            let mut max_val = f32::MIN;
            for c in 0..classes {
                let val = if layout_ntc {
                    let idx = ((batch_idx * seq_len + t) * classes) + c;
                    data[idx]
                } else {
                    let idx = ((batch_idx * classes + c) * seq_len) + t;
                    data[idx]
                };
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }

            if max_idx == 0 {
                prev = max_idx as i32;
                continue;
            }
            if prev == max_idx as i32 {
                continue;
            }
            let dict_idx = max_idx - 1;
            if dict_idx < dict.len() {
                text.push_str(&dict[dict_idx]);
                probs.push(max_val);
            }
            prev = max_idx as i32;
        }

        let score = if probs.is_empty() {
            0.0
        } else {
            probs.iter().sum::<f32>() / probs.len() as f32
        };
        (text, score)
    }
}

#[cfg(not(feature = "openvino"))]
mod imp {
    use super::*;

    pub struct OpenVinoOcrPipeline;

    impl OpenVinoOcrPipeline {
        pub fn new(_model: &OpenVinoOcrModelConfig, _pipeline: &OpenVinoPipelineConfig) -> Result<Self, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn run_image(&mut self, _data: *const u8, _len: usize) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn line_count(&self) -> Result<usize, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn write_lines(&self, _dst: *mut SdkOcrLine, _cap: usize, _out_written: *mut usize) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn text_bytes(&self) -> Result<usize, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn write_text(&self, _dst: *mut i8, _cap: usize, _out_written: *mut usize) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }

        pub fn struct_json_bytes(&self) -> Result<usize, SdkErr> {
            set_last_error("openvino feature not enabled");
            Err(SdkErr::Unsupported)
        }

        pub fn write_struct_json(&self, _dst: *mut i8, _cap: usize, _out_written: *mut usize) -> SdkErr {
            set_last_error("openvino feature not enabled");
            SdkErr::Unsupported
        }
    }
}

pub use imp::OpenVinoOcrPipeline;
