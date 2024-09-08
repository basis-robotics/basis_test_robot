/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include "basis/core/logging/macros.h"
#include <cstdint>
#include <onnxruntime_c_api.h>
#include <resnet_demo.h>

#include <onnxruntime_cxx_api.h>

#include <cuda_runtime_api.h>

#include <prepare_image.h>

using namespace unit::resnet_demo;

// onnx forces a global logger, but sessions should use their own
Ort::Env global_ort_env(ORT_LOGGING_LEVEL_WARNING, "global_onnx_runtime");

Inference::Inference(const char *model_path) {
  const auto &api = Ort::GetApi();


  
  Ort::SessionOptions session_options;

  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#if 0
  // tensorrt
  OrtTensorRTProviderOptionsV2 *tensorrt_options;

  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
  // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options
  const char *options_names[] = {
      "device_id",
      "trt_max_workspace_size",
      "trt_engine_cache_enable",
      "trt_engine_cache_path",
      "trt_timing_cache_enable",
      "trt_timing_cache_path",
  };
  const char *options_values[] = {"0", "2147483648", "1", "/tmp/onnx_cache", "1", "/tmp/onnx_cache"};

  Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, options_names, options_values, 0));

  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
      tensorrt_options, api.ReleaseTensorRTProviderOptions);

  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
      static_cast<OrtSessionOptions *>(session_options), rel_trt_options.get()));
#endif
#if 1
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

// wget https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx?raw=true -O yolov4.onnx maybe

  //  tensorrt_options->device_id = 0; // Choose GPU device (0 for default)
  //   tensorrt_options->has_user_compute_stream = 0; // No custom CUDA stream

  //   // Set maximum workspace size (in bytes) for TensorRT
  //   tensorrt_options->trt_max_workspace_size = 1ULL << 30; // 1GB workspace size for TensorRT

  //   // Enable FP16 precision mode for TensorRT if supported (optional)
  //   tensorrt_options->trt_fp16_enable = 1; // 1 to enable, 0 to disable

  std::cout << "Running ORT TRT EP with default provider options" << std::endl;

  session = std::make_unique<Ort::Session>(global_ort_env, model_path, session_options);
  DetectInputsOutputs();
}

void Inference::DetectInputsOutputs() {
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  {
    // print number of model input nodes
    const size_t num_input_nodes = session->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char *> input_node_names;
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    std::vector<std::vector<int64_t>> input_node_dims;

    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
      // print input node names
      auto input_name = session->GetInputNameAllocated(i, allocator);
      std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
      input_node_names.push_back(input_name.get());
      input_names_ptr.push_back(std::move(input_name));

      // print input node types
      auto type_info = session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType type = tensor_info.GetElementType();
      std::cout << "Input " << i << " : type = " << type << std::endl;

      // print input shapes/dims
      input_node_dims.push_back(tensor_info.GetShape());
      auto dims = input_node_dims.back();
      std::cout << "Input " << i << " : num_dims = " << dims.size() << '\n';
      for (size_t j = 0; j < dims.size(); j++) {
        std::cout << "Input " << i << " : dim[" << j << "] =" << dims[j] << '\n';
      }
      std::cout << std::flush;
    }
  }

  {
    // print number of model output nodes
    const size_t num_output_nodes = session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<const char *> output_node_names;
    output_names_ptr.reserve(num_output_nodes);
    output_node_names.reserve(num_output_nodes);
    std::vector<std::vector<int64_t>> output_node_dims;

    std::cout << "Number of outputs = " << num_output_nodes << std::endl;

    // iterate over all output nodes
    for (size_t i = 0; i < num_output_nodes; i++) {
      // print output node names
      auto output_name = session->GetOutputNameAllocated(i, allocator);
      std::cout << "output " << i << " : name =" << output_name.get() << std::endl;
      output_node_names.push_back(output_name.get());
      output_names_ptr.push_back(std::move(output_name));

      // print input node types
      auto type_info = session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType type = tensor_info.GetElementType();
      std::cout << "output " << i << " : type = " << type << std::endl;

      // print output shapes/dims
      output_node_dims.push_back(tensor_info.GetShape());
      auto dims = output_node_dims.back();
      std::cout << "output " << i << " : num_dims = " << dims.size() << '\n';
      for (size_t j = 0; j < dims.size(); j++) {
        std::cout << "output " << i << " : dim[" << j << "] =" << dims[j] << '\n';
      }
      std::cout << std::flush;
    }
  }
}

void Inference::Infer(const image_conversion::CudaManagedImage &image) {
  // prepare image
  // TODO: add float type to CudaManagedImage
  float *buffer;
  const int buffer_size = image.width * image.height * 3 * sizeof(float);
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&buffer, buffer_size));
  RGBToTensor((const unsigned char *)image.buffer, buffer, image.width, image.height);
  CheckCudaError();


  // Create an Ort tensor from the input data
  Ort::MemoryInfo cuda_memory_info("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::MemoryInfo cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create input tensor shape: {1, 3, 224, 224}
  std::array<int64_t, 4> inputShape = {1, 3, image.width, image.height};

  Ort::Value pixel_values_tensor =
     Ort::Value::CreateTensor<float>(cuda_memory_info, buffer, buffer_size, inputShape.data(), inputShape.size());

  // char* cpu_buffer = (char*)malloc(buffer_size);
  // cudaMemcpy(cpu_buffer, buffer, buffer_size, cudaMemcpyDeviceToHost);
  // Ort::Value pixel_values_tensor =
  //     Ort::Value::CreateTensor<float>(cpu_memory_info, (float*)cpu_buffer, buffer_size, inputShape.data(), inputShape.size());

  std::array<int64_t, 3> masks_shape = {1, 64, 64};

  // Create zero mask data (int64_t) for pixel_mask as a placeholder
  size_t num_elements = 64 * 64;
  std::vector<int64_t> pixel_mask_data(num_elements, 0); // Initialize all elements with zeros

  Ort::Value masks_tensor = Ort::Value::CreateTensor<int64_t>(
      cpu_memory_info, pixel_mask_data.data(), pixel_mask_data.size(), masks_shape.data(), masks_shape.size());

  const char *input_names[] = {"pixel_values", "pixel_mask"};
  const char *output_names[] = {"logits", "pred_boxes"};

  std::array<Ort::Value, 2> inputs = {std::move(pixel_values_tensor), std::move(masks_tensor)};

  auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &inputs.at(0), 1, output_names, 2);

  // https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/
  /*
  
    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    assert(abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < 5; i++) {
      std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
    }
    std::cout << std::flush;
    */
  CUDA_SAFE_CALL_NO_SYNC(cudaFree(buffer));
}

resnet_demo::resnet_demo(std::optional<std::string> name_override) : unit::resnet_demo::Base(name_override) {

  //inference = std::make_unique<Inference>("/basis_test_robot/docker/model.onnx");
  inference = std::make_unique<Inference>("/basis_test_robot/docker/test/model.onnx");
  

#if 0
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

   // print number of model input nodes
  const size_t num_input_nodes = session->GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_dims;

  std::cout << "Number of inputs = " << num_input_nodes << std::endl;

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session->GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

    // print input node types
    auto type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input " << i << " : type = " << type << std::endl;

    // print input shapes/dims
    input_node_dims.push_back(tensor_info.GetShape());
    auto dims = input_node_dims.back();
    std::cout << "Input " << i << " : num_dims = " << dims.size() << '\n';
    for (size_t j = 0; j < dims.size(); j++) {
      std::cout << "Input " << i << " : dim[" << j << "] =" << dims[j] << '\n';
    }
    std::cout << std::flush;
  }
#endif
}
OnRGB::Output resnet_demo::OnRGB(const OnRGB::Input &input) {
  BASIS_LOG_INFO("Got RGB");
  inference->Infer(*input.camera_rgb_cuda.get());
  return {};
}
