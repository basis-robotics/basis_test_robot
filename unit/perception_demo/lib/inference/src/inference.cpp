#include <nppdefs.h>
#include <onnxruntime_cxx_api.h>

#include <cuda_runtime_api.h>

#include <coco_labels.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
#include <prepare_image.h>

#include <inference.h>

#include <iostream>

#include <foxglove/RawImage.pb.h>
#include <spdlog/spdlog.h>

// onnx forces a global logger, but sessions should use their own
Ort::Env global_ort_env(ORT_LOGGING_LEVEL_WARNING, "global_onnx_runtime");

Inference::Inference(const std::filesystem::path &model_dir, const std::string_view platform, InferenceRunMode mode,
                     int inference_width, int inference_height)
    : inference_width(inference_width), inference_height(inference_height) {
  const auto &api = Ort::GetApi();

  bool enable_fp16 = false;

  std::string cache_key = fmt::format("{}_{}_{}", platform, inference_width, inference_height);
  if (enable_fp16) {
    cache_key += "_fp16";
  }
  const std::filesystem::path cache_dir = model_dir / cache_key;
  if (mode == InferenceRunMode::DUMP_CACHE) {
    SPDLOG_INFO("Will write model to {}", cache_dir.c_str());
    std::filesystem::create_directory(cache_dir);
  }
  const std::filesystem::path model_path =
      mode == InferenceRunMode::INFERENCE ? cache_dir / "model_ctx.onnx" : model_dir / "model.onnx";

  Ort::SessionOptions session_options;

  session_options.SetIntraOpNumThreads(4);
  session_options.SetInterOpNumThreads(4);
  session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // tensorrt
  OrtTensorRTProviderOptionsV2 *tensorrt_options;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

  std::vector<std::pair<std::string, std::string>> args = {
      {"device_id", "0"},
      {"trt_max_workspace_size", "2147483648"},
      {"trt_engine_cache_enable", "1"},
      {"trt_dump_ep_context_model", mode == InferenceRunMode::DUMP_CACHE ? "1" : "0"},
      {"trt_ep_context_file_path", cache_dir},
      {"trt_engine_cache_path", "./cache"},
      {"trt_fp16_enable", enable_fp16 ? "1" : "0"},
      {"trt_layer_norm_fp32_fallback", enable_fp16 ? "1" : "0"},

      // {"trt_int8_enable", "1"},
      // {"trt_dla_enable", "1"},
      // {"trt_dla_core", "0"},
  };

  std::vector<const char *> option_names;
  std::vector<const char *> option_values;
  for (const auto &arg : args) {
    option_names.push_back(arg.first.c_str());
    option_values.push_back(arg.second.c_str());
  }
  Ort::ThrowOnError(
      api.UpdateTensorRTProviderOptions(tensorrt_options, option_names.data(), option_values.data(), args.size()));

  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
      tensorrt_options, api.ReleaseTensorRTProviderOptions);

  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
      static_cast<OrtSessionOptions *>(session_options), rel_trt_options.get()));

#if 0
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // cuda
  OrtCUDAProviderOptionsV2 *cuda_options;

  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options
  
  //Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, options_names, options_values, 0));

  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(
      cuda_options, api.ReleaseCUDAProviderOptions);

  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
      static_cast<OrtSessionOptions *>(session_options), rel_cuda_options.get()));
#endif
  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  session = std::make_unique<Ort::Session>(global_ort_env, model_path.c_str(), session_options);
  DetectInputsOutputs();

  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&inference_buffer, BufferSize()));
  CUDA_SAFE_CALL_NO_SYNC(cudaMemset(inference_buffer, BufferSize(), 0));
}

Inference::~Inference() { CUDA_SAFE_CALL_NO_SYNC(cudaFree(inference_buffer)); }

template <typename T> void PrintTensorShape(const T &tensor_info, int i, std::string_view desc) {
  ONNXTensorElementDataType type = tensor_info.GetElementType();
  std::cout << desc << " " << i << " : type = " << type << std::endl;

  // print shapes/dims
  auto dims = tensor_info.GetShape();
  std::cout << desc << " " << i << " : num_dims = " << dims.size() << '\n';
  for (size_t j = 0; j < dims.size(); j++) {
    std::cout << desc << " " << i << " : dim[" << j << "] = " << dims[j] << '\n';
  }
  std::cout << std::flush;
}

void Inference::DetectInputsOutputs() {
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  {
    // iterate over all output nodes
    for (size_t i = 0; i < session->GetInputCount(); i++) {
      // print output node names
      auto input_name = session->GetInputNameAllocated(i, allocator);
      std::cout << "input " << i << " : name =" << input_name.get() << std::endl;

      auto type_info = session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      PrintTensorShape(tensor_info, i, "input");
    }
  }

  {
    // iterate over all output nodes
    for (size_t i = 0; i < session->GetOutputCount(); i++) {
      // print output node names
      auto output_name = session->GetOutputNameAllocated(i, allocator);
      std::cout << "output " << i << " : name =" << output_name.get() << std::endl;
      auto type_info = session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      PrintTensorShape(tensor_info, i, "input");
    }
  }
}
// https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c
static void softmax(float *input, size_t input_len) {
  assert(input);

  float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}

std::vector<Detection> InferenceDetrResnet::Infer(const image_conversion::CudaManagedImage &image) {
  // Do a really dumb thing and resize to match what the model expects
  // we really should be padding and masking instead :(
  if (last_input_time > image.time) {
    SPDLOG_ERROR("{} > {}", last_input_time.ToSeconds(), image.time.ToSeconds());
  }
  last_input_time = image.time;

  if (image.width > inference_width || image.height > inference_height) {
    SPDLOG_ERROR("Invalid inference size: {}x{} > {}x{}", image.width, image.height, inference_width, inference_height);
    return {};
  }

  RGBToTensor((const unsigned char *)image.buffer, image.width, image.height, image.StepSize(), inference_buffer,
              inference_width, inference_height);

  image_conversion::CheckCudaError();

  auto dets = Infer();

  return dets;
}

Detection CreateDetection(const char *label, float bb_cx, float bb_cy, float bb_w, float bb_h, float image_width,
                          float image_height) {
  const float bb_w_2 = bb_w / 2;
  const float bb_h_2 = bb_h / 2;
  float points_float[4][2] = {
      {bb_cx - bb_w_2, bb_cy - bb_h_2},
      {bb_cx + bb_w_2, bb_cy - bb_h_2},
      {bb_cx + bb_w_2, bb_cy + bb_h_2},
      {bb_cx - bb_w_2, bb_cy + bb_h_2},
  };

  return Detection(label, std::array<std::array<int, 2>, 4>{
                     std::array<int, 2>{int(points_float[0][0] * image_width), int(points_float[0][1] * image_height)},
                     {int(points_float[1][0] * image_width), int(points_float[1][1] * image_height)},
                     {int(points_float[2][0] * image_width), int(points_float[2][1] * image_height)},
                     {int(points_float[3][0] * image_width), int(points_float[3][1] * image_height)},
                 });
}

std::vector<Detection> InferenceDetrResnet::Infer() {
  std::array<int64_t, 4> inputShape = {1, 3, inference_height, inference_width};
  // Create an Ort tensor from the input data

  Ort::MemoryInfo cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensor shape: {1, 3, 224, 224}
  // std::cout << resize_width << "x" << resize_height << std::endl;
  // std::cout << buffer_size << " =?= " << 1 * 3 * resize_width * resize_height * sizeof(float) << std::endl;
  Ort::Value pixel_values_tensor = CreateInputInferenceTensor();

  std::array<int64_t, 3> masks_shape = {1, 64, 64};

  // Create zero mask data (int64_t) for pixel_mask as a placeholder
  size_t num_elements = 64 * 64;
  std::vector<int64_t> pixel_mask_data(num_elements, 1);

  // todo: hardcoded
  int mask_boundary = (64.0 * 720.0 / 800.0);
  for (int i = mask_boundary; i < 64; i++) {
    for (int j = 0; j < 64; j++) {
      pixel_mask_data[i * 64 + j] = 0;
    }
  }

  Ort::Value masks_tensor = Ort::Value::CreateTensor<int64_t>(
      cpu_memory_info, pixel_mask_data.data(), pixel_mask_data.size(), masks_shape.data(), masks_shape.size());

  const char *input_names[] = {"pixel_values", "pixel_mask"};
  const char *output_names[] = {"logits", "pred_boxes"};

  std::array<Ort::Value, 2> inputs = {std::move(pixel_values_tensor), std::move(masks_tensor)};

  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &inputs.at(0), 2, output_names, 2);

  float *logits = output_tensors[0].GetTensorMutableData<float>();
  const float *all_bboxes = output_tensors[1].GetTensorData<float>();

  const auto &logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  const int batch_size = logits_shape[0];
  const int num_queries = logits_shape[1];
  const int label_count = logits_shape[2];

  const auto &bboxes_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

  std::vector<Detection> out;

  assert(batch_size == 1);
  for (int object_idx = 0; object_idx < num_queries; object_idx++) {
    float *labels = logits + object_idx * label_count;
    const float *bbox = all_bboxes + object_idx * 4;

    softmax(labels, label_count);
    float largest_value = 0;
    size_t largest_index = label_count - 1;
    for (int label_idx = 0; label_idx < label_count; label_idx++) {
      if (labels[label_idx] > largest_value) {
        largest_value = labels[label_idx];
        largest_index = label_idx;
      }
    }

    if (largest_index != label_count - 1) {
      if (largest_value > 0.9f) {
        const char *label = coco_labels[largest_index];

        const float bb_cx = bbox[0];
        const float bb_cy = bbox[1];
        const float bb_w_2 = bbox[2] / 2;
        const float bb_h_2 = bbox[3] / 2;
        float points_float[4][2] = {
            {bb_cx - bb_w_2, bb_cy - bb_h_2},
            {bb_cx + bb_w_2, bb_cy - bb_h_2},
            {bb_cx + bb_w_2, bb_cy + bb_h_2},
            {bb_cx - bb_w_2, bb_cy + bb_h_2},
        };

        out.emplace_back(label,
                         std::array<std::array<int, 2>, 4>{
                             std::array<int, 2>{int(points_float[0][0] * inference_width),
                                                int(points_float[0][1] * inference_height)},
                             {int(points_float[1][0] * inference_width), int(points_float[1][1] * inference_height)},
                             {int(points_float[2][0] * inference_width), int(points_float[2][1] * inference_height)},
                             {int(points_float[3][0] * inference_width), int(points_float[3][1] * inference_height)},
                         });
        // TODO: clipping
      }
    }
  }
  return out;
}

std::array<std::shared_ptr<foxglove::RawImage>, 3> Inference::DumpInferenceBuffer() {
  std::array<std::shared_ptr<foxglove::RawImage>, 3> out;

  const int single_channel_size = BufferSize() / 3;
  const timespec ts = last_input_time.ToTimespec();

  for (int channel = 0; channel < 3; channel++) {
    auto image_msg = std::make_shared<foxglove::RawImage>();

    image_msg->mutable_timestamp()->set_seconds(ts.tv_sec);
    image_msg->mutable_timestamp()->set_nanos(ts.tv_nsec);
    image_msg->set_frame_id("inference");
    image_msg->set_encoding("32FC1");

    image_msg->set_width(inference_width);
    image_msg->set_height(inference_height);
    image_msg->set_step(inference_width * sizeof(float));
    const auto &data = image_msg->mutable_data();

    data->resize(single_channel_size);

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(data->data(), (unsigned char *)inference_buffer + single_channel_size * channel,
                                      single_channel_size, cudaMemcpyDeviceToHost));

    out[channel] = image_msg;
  }

  return out;
}

Ort::Value Inference::CreateInputInferenceTensor() {
  Ort::MemoryInfo cuda_memory_info("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  const std::array<int64_t, 4> inputShape = {1, 3, inference_height, inference_width};
  return Ort::Value::CreateTensor<float>(cuda_memory_info, inference_buffer, BufferSize(), inputShape.data(),
                                         inputShape.size());
}

InferenceDetrResnet::InferenceDetrResnet(const std::filesystem::path &model_dir, const std::string_view platform,
                                         InferenceRunMode mode)
    : Inference(model_dir, platform, mode, 1280, 800) {}

InferenceYoloV9::InferenceYoloV9(const std::filesystem::path &model_dir, const std::string_view platform,
                                 InferenceRunMode mode)
    : Inference(model_dir, platform, mode, 640, 640) {
  resize_buffer = nppiMalloc_8u_C3(inference_width, inference_height, &resize_step);
}

InferenceYoloV9::~InferenceYoloV9() { nppiFree(resize_buffer); }
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
std::vector<Detection> InferenceYoloV9::Infer(const image_conversion::CudaManagedImage &image) {
  if (last_input_time > image.time) {
    SPDLOG_ERROR("{} > {}", last_input_time.ToSeconds(), image.time.ToSeconds());
  }
  last_input_time = image.time;

  NppStatus error = nppiResize_8u_C3R(
      (const unsigned char *)image.buffer, image.StepSize(), {image.width, image.height},
      {0, 0, image.width, image.height}, resize_buffer, resize_step,
      // Resize, preserving aspect ratio
      {inference_width, inference_height * image.height / image.width},
      {0, 0, inference_width, inference_height * image.height / image.width}, NppiInterpolationMode::NPPI_INTER_LINEAR);
  if (error) {
    SPDLOG_ERROR("nppiResize_8u_C3R fail {}", (int)error);
    return {};
  }

  RGBToTensor((const unsigned char *)resize_buffer, inference_width, inference_height, resize_step, inference_buffer,
              inference_width, inference_height);

  image_conversion::CheckCudaError();

  Ort::Value pixel_values_tensor = CreateInputInferenceTensor();

  const char *input_names[] = {"images"};
  const char *output_names[] = {"output0"};

  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &pixel_values_tensor, 1, output_names, 1);

  float *output_tensor = output_tensors[0].GetTensorMutableData<float>();

  float confidence_threshold = 0.5f;
  int num_anchors = 8400;
  int num_features = 79;
  std::vector<Detection> out;

  for (int anchor = 0; anchor < num_anchors; ++anchor) {
    // Calculate indices in memory
    int box_offset = anchor;
    int class_start_idx = 4 * num_anchors + box_offset; // Class probabilities start

    // Access objectness score

    // Access class probabilities (apply sigmoid if needed)
    float max_class_prob = 0.0f;
    int max_class_id = -1;
    for (int cls = 0; cls < num_features; ++cls) {
      float class_prob = output_tensor[class_start_idx + cls * num_anchors];
      if (class_prob > max_class_prob) {
        max_class_prob = class_prob;
        max_class_id = cls;
      }
    }

    // Compute final confidence score
    float final_confidence = max_class_prob;

    // Filter based on confidence threshold
    if (final_confidence > confidence_threshold) {
      // Process the detection (e.g., get bounding box, class id)
      float x = output_tensor[box_offset];
      float y = output_tensor[box_offset + num_anchors];
      float w = output_tensor[box_offset + 2 * num_anchors];
      float h = output_tensor[box_offset + 3 * num_anchors];

      // Now you have high-confidence detections: x, y, w, h, max_class_id, final_confidence
      out.emplace_back(CreateDetection(yolo_labels[max_class_id], x / inference_width, y/ inference_height, w / inference_width, h / inference_height, 1280, 1280));
    }
  }

  return out;
}
