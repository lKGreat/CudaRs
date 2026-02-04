#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "src/pipelines/ocr/pipeline.h"
#include "src/pipelines/ocr/result.h"
#include "src/utils/utility.h"

extern "C" {

struct PaddleOcrInitOptions {
  const char *doc_orientation_model_name;
  const char *doc_orientation_model_dir;
  const char *doc_unwarping_model_name;
  const char *doc_unwarping_model_dir;
  const char *text_detection_model_name;
  const char *text_detection_model_dir;
  const char *textline_orientation_model_name;
  const char *textline_orientation_model_dir;
  const char *text_recognition_model_name;
  const char *text_recognition_model_dir;
  const char *lang;
  const char *ocr_version;
  const char *vis_font_dir;
  const char *device;
  const char *precision;
  const char *text_det_limit_type;
  const char *paddlex_config_yaml;

  int32_t textline_orientation_batch_size;
  int32_t text_recognition_batch_size;
  int32_t use_doc_orientation_classify;
  int32_t use_doc_unwarping;
  int32_t use_textline_orientation;
  int32_t text_det_limit_side_len;
  int32_t text_det_max_side_limit;
  int32_t enable_mkldnn;
  int32_t mkldnn_cache_capacity;
  int32_t cpu_threads;
  int32_t thread_num;
  int32_t enable_struct_json;

  float text_det_thresh;
  float text_det_box_thresh;
  float text_det_unclip_ratio;
  float text_rec_score_thresh;

  int32_t text_det_input_shape[4];
  int32_t text_rec_input_shape[4];
  int32_t text_det_input_shape_len;
  int32_t text_rec_input_shape_len;
};

struct PaddleOcrLine {
  float points[8];
  float score;
  int32_t cls_label;
  float cls_score;
  uint32_t text_offset;
  uint32_t text_len;
};

struct PaddleOcrHandle;

int paddleocr_create(const PaddleOcrInitOptions *options,
                     PaddleOcrHandle **out_handle);
int paddleocr_destroy(PaddleOcrHandle *handle);
int paddleocr_run_image(PaddleOcrHandle *handle, const uint8_t *data,
                        size_t len);
int paddleocr_get_line_count(PaddleOcrHandle *handle, size_t *out_count);
int paddleocr_write_lines(PaddleOcrHandle *handle, PaddleOcrLine *dst,
                          size_t cap, size_t *out_written);
int paddleocr_get_text_bytes(PaddleOcrHandle *handle, size_t *out_bytes);
int paddleocr_write_text(PaddleOcrHandle *handle, char *dst, size_t cap,
                         size_t *out_written);
int paddleocr_get_struct_json_bytes(PaddleOcrHandle *handle,
                                    size_t *out_bytes);
int paddleocr_write_struct_json(PaddleOcrHandle *handle, char *dst, size_t cap,
                                size_t *out_written);
const char *paddleocr_last_error(size_t *out_len);
}

struct PaddleOcrHandle {
  std::unique_ptr<_OCRPipeline> pipeline;
  std::vector<OCRPipelineResult> last_results;
  std::vector<PaddleOcrLine> last_lines;
  std::string last_texts;
  std::string last_struct_json;
  bool enable_struct_json = false;
};

namespace {

thread_local std::string g_last_error;

void SetLastError(const std::string &msg) { g_last_error = msg; }

bool HasStr(const char *value) { return value && value[0] != '\0'; }

absl::optional<std::string> OptStr(const char *value) {
  if (!HasStr(value)) {
    return absl::nullopt;
  }
  return std::string(value);
}

absl::optional<int> OptI32(int32_t value) {
  if (value == std::numeric_limits<int32_t>::min()) {
    return absl::nullopt;
  }
  return static_cast<int>(value);
}

absl::optional<float> OptF32(float value) {
  if (std::isnan(value)) {
    return absl::nullopt;
  }
  return value;
}

std::string TempFilePath(const char *ext) {
  auto base = std::filesystem::temp_directory_path();
  auto name = std::string("cudars_ocr_") +
              std::to_string(std::chrono::high_resolution_clock::now()
                                 .time_since_epoch()
                                 .count()) +
              (ext ? ext : ".bin");
  return (base / name).string();
}

const char *DetectExt(const uint8_t *data, size_t len) {
  if (!data || len < 4) {
    return ".img";
  }
  if (len >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E &&
      data[3] == 0x47) {
    return ".png";
  }
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
    return ".jpg";
  }
  return ".img";
}

void BuildLinesFromResult(PaddleOcrHandle *handle) {
  handle->last_lines.clear();
  handle->last_texts.clear();

  uint32_t offset = 0;
  for (const auto &res : handle->last_results) {
    size_t count = res.rec_texts.size();
    for (size_t i = 0; i < count; ++i) {
      PaddleOcrLine line{};
      const auto &text = res.rec_texts[i];
      line.text_offset = offset;
      line.text_len = static_cast<uint32_t>(text.size());
      offset += line.text_len;
      handle->last_texts.append(text);

      line.score = (i < res.rec_scores.size()) ? res.rec_scores[i] : 0.0f;
      line.cls_label =
          (i < res.textline_orientation_angles.size())
              ? res.textline_orientation_angles[i]
              : 0;
      line.cls_score = 0.0f;

      const std::vector<cv::Point2f> *poly = nullptr;
      if (i < res.rec_polys.size()) {
        poly = &res.rec_polys[i];
      } else if (i < res.dt_polys.size()) {
        poly = &res.dt_polys[i];
      }
      if (poly && poly->size() >= 4) {
        for (int p = 0; p < 4; ++p) {
          line.points[p * 2] = (*poly)[p].x;
          line.points[p * 2 + 1] = (*poly)[p].y;
        }
      }

      handle->last_lines.push_back(line);
    }
  }
}

void BuildStructJson(PaddleOcrHandle *handle) {
  handle->last_struct_json.clear();
  handle->last_struct_json.push_back('[');
  bool first = true;
  for (const auto &res : handle->last_results) {
    OCRResult result(res);
    std::string tmp_path = TempFilePath(".json");
    result.SaveToJson(tmp_path);
    std::ifstream file(tmp_path, std::ios::binary);
    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    if (!json.empty()) {
      if (!first) {
        handle->last_struct_json.push_back(',');
      }
      handle->last_struct_json.append(json);
      first = false;
    }
  }
  handle->last_struct_json.push_back(']');
}

} // namespace

extern "C" {

const char *paddleocr_last_error(size_t *out_len) {
  if (out_len) {
    *out_len = g_last_error.size();
  }
  return g_last_error.c_str();
}

int paddleocr_create(const PaddleOcrInitOptions *options,
                     PaddleOcrHandle **out_handle) {
  if (!options || !out_handle) {
    SetLastError("options or out_handle is null");
    return -1;
  }

  OCRPipelineParams params;
  params.doc_orientation_classify_model_name =
      OptStr(options->doc_orientation_model_name);
  params.doc_orientation_classify_model_dir =
      OptStr(options->doc_orientation_model_dir);
  params.doc_unwarping_model_name = OptStr(options->doc_unwarping_model_name);
  params.doc_unwarping_model_dir = OptStr(options->doc_unwarping_model_dir);
  params.text_detection_model_name = OptStr(options->text_detection_model_name);
  params.text_detection_model_dir = OptStr(options->text_detection_model_dir);
  params.textline_orientation_model_name =
      OptStr(options->textline_orientation_model_name);
  params.textline_orientation_model_dir =
      OptStr(options->textline_orientation_model_dir);
  params.text_recognition_model_name =
      OptStr(options->text_recognition_model_name);
  params.text_recognition_model_dir =
      OptStr(options->text_recognition_model_dir);
  params.lang = OptStr(options->lang);
  params.ocr_version = OptStr(options->ocr_version);
  params.vis_font_dir = OptStr(options->vis_font_dir);
  params.device = OptStr(options->device);
  if (HasStr(options->precision)) {
    params.precision = options->precision;
  }
  params.textline_orientation_batch_size =
      OptI32(options->textline_orientation_batch_size);
  params.text_recognition_batch_size =
      OptI32(options->text_recognition_batch_size);
  params.use_doc_orientation_classify =
      OptI32(options->use_doc_orientation_classify).has_value()
          ? absl::optional<bool>(
                OptI32(options->use_doc_orientation_classify).value() != 0)
          : absl::nullopt;
  params.use_doc_unwarping =
      OptI32(options->use_doc_unwarping).has_value()
          ? absl::optional<bool>(OptI32(options->use_doc_unwarping).value() !=
                                 0)
          : absl::nullopt;
  params.use_textline_orientation =
      OptI32(options->use_textline_orientation).has_value()
          ? absl::optional<bool>(
                OptI32(options->use_textline_orientation).value() != 0)
          : absl::nullopt;
  params.text_det_limit_side_len = OptI32(options->text_det_limit_side_len);
  params.text_det_limit_type = OptStr(options->text_det_limit_type);
  params.text_det_thresh = OptF32(options->text_det_thresh);
  params.text_det_box_thresh = OptF32(options->text_det_box_thresh);
  params.text_det_unclip_ratio = OptF32(options->text_det_unclip_ratio);
  params.text_rec_score_thresh = OptF32(options->text_rec_score_thresh);

  if (options->text_det_input_shape_len > 0) {
    std::vector<int> shape;
    for (int i = 0; i < options->text_det_input_shape_len && i < 4; ++i) {
      shape.push_back(options->text_det_input_shape[i]);
    }
    params.text_det_input_shape = shape;
  }
  if (options->text_rec_input_shape_len > 0) {
    std::vector<int> shape;
    for (int i = 0; i < options->text_rec_input_shape_len && i < 4; ++i) {
      shape.push_back(options->text_rec_input_shape[i]);
    }
    params.text_rec_input_shape = shape;
  }

  if (HasStr(options->paddlex_config_yaml)) {
    params.paddlex_config =
        Utility::PaddleXConfigVariant(options->paddlex_config_yaml);
  }

  if (options->enable_mkldnn != std::numeric_limits<int32_t>::min()) {
    params.enable_mkldnn = options->enable_mkldnn != 0;
  }
  if (options->mkldnn_cache_capacity != std::numeric_limits<int32_t>::min()) {
    params.mkldnn_cache_capacity = options->mkldnn_cache_capacity;
  }
  if (options->cpu_threads != std::numeric_limits<int32_t>::min()) {
    params.cpu_threads = options->cpu_threads;
  }
  if (options->thread_num != std::numeric_limits<int32_t>::min()) {
    params.thread_num = options->thread_num;
  }

  try {
    auto pipeline = std::unique_ptr<_OCRPipeline>(new _OCRPipeline(params));
    auto handle = new PaddleOcrHandle();
    handle->pipeline = std::move(pipeline);
    handle->enable_struct_json = (options->enable_struct_json != 0);
    *out_handle = handle;
    return 0;
  } catch (const std::exception &ex) {
    SetLastError(ex.what());
    return -1;
  } catch (...) {
    SetLastError("unknown error in paddleocr_create");
    return -1;
  }
}

int paddleocr_destroy(PaddleOcrHandle *handle) {
  if (!handle) {
    return -1;
  }
  delete handle;
  return 0;
}

int paddleocr_run_image(PaddleOcrHandle *handle, const uint8_t *data,
                        size_t len) {
  if (!handle || !data || len == 0) {
    SetLastError("invalid handle or input data");
    return -1;
  }
  const char *ext = DetectExt(data, len);
  std::string path = TempFilePath(ext);
  {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
      SetLastError("failed to open temp file for input");
      return -1;
    }
    file.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(len));
  }

  try {
    std::vector<std::string> inputs = {path};
    handle->pipeline->Predict(inputs);
    handle->last_results = handle->pipeline->PipelineResult();
    BuildLinesFromResult(handle);
    if (handle->enable_struct_json) {
      BuildStructJson(handle);
    } else {
      handle->last_struct_json.clear();
    }
  } catch (const std::exception &ex) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    SetLastError(ex.what());
    return -1;
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    SetLastError("unknown error in paddleocr_run_image");
    return -1;
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  return 0;
}

int paddleocr_get_line_count(PaddleOcrHandle *handle, size_t *out_count) {
  if (!handle || !out_count) {
    SetLastError("handle or out_count is null");
    return -1;
  }
  *out_count = handle->last_lines.size();
  return 0;
}

int paddleocr_write_lines(PaddleOcrHandle *handle, PaddleOcrLine *dst,
                          size_t cap, size_t *out_written) {
  if (!handle || !dst || !out_written) {
    SetLastError("handle or output pointers are null");
    return -1;
  }
  size_t to_copy = std::min(cap, handle->last_lines.size());
  if (to_copy > 0) {
    std::memcpy(dst, handle->last_lines.data(),
                to_copy * sizeof(PaddleOcrLine));
  }
  *out_written = to_copy;
  if (to_copy < handle->last_lines.size()) {
    SetLastError("line buffer too small");
    return -1;
  }
  return 0;
}

int paddleocr_get_text_bytes(PaddleOcrHandle *handle, size_t *out_bytes) {
  if (!handle || !out_bytes) {
    SetLastError("handle or out_bytes is null");
    return -1;
  }
  *out_bytes = handle->last_texts.size();
  return 0;
}

int paddleocr_write_text(PaddleOcrHandle *handle, char *dst, size_t cap,
                         size_t *out_written) {
  if (!handle || !dst || !out_written) {
    SetLastError("handle or output pointers are null");
    return -1;
  }
  size_t to_copy = std::min(cap, handle->last_texts.size());
  if (to_copy > 0) {
    std::memcpy(dst, handle->last_texts.data(), to_copy);
  }
  *out_written = to_copy;
  if (to_copy < handle->last_texts.size()) {
    SetLastError("text buffer too small");
    return -1;
  }
  return 0;
}

int paddleocr_get_struct_json_bytes(PaddleOcrHandle *handle,
                                    size_t *out_bytes) {
  if (!handle || !out_bytes) {
    SetLastError("handle or out_bytes is null");
    return -1;
  }
  *out_bytes = handle->last_struct_json.size();
  return 0;
}

int paddleocr_write_struct_json(PaddleOcrHandle *handle, char *dst, size_t cap,
                                size_t *out_written) {
  if (!handle || !dst || !out_written) {
    SetLastError("handle or output pointers are null");
    return -1;
  }
  size_t to_copy = std::min(cap, handle->last_struct_json.size());
  if (to_copy > 0) {
    std::memcpy(dst, handle->last_struct_json.data(), to_copy);
  }
  *out_written = to_copy;
  if (to_copy < handle->last_struct_json.size()) {
    SetLastError("struct json buffer too small");
    return -1;
  }
  return 0;
}

} // extern "C"
