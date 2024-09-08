/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include "basis/core/logging/macros.h"
#include <resnet_demo.h>

#include <onnxruntime_cxx_api.h>

#include <cuda_runtime_api.h>

#include <prepare_image.h>

using namespace unit::resnet_demo;

// onnx forces a global logger, but sessions should use their own
Ort::Env global_ort_env(ORT_LOGGING_LEVEL_WARNING, "global_onnx_runtime");

Inference::Inference(const char *model_path) {
  const auto &api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2 *tensorrt_options;

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
      tensorrt_options, api.ReleaseTensorRTProviderOptions);
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
      static_cast<OrtSessionOptions *>(session_options), rel_trt_options.get()));

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
  float* buffer;
  CUDA_SAFE_CALL_NO_SYNC(cudaMallocManaged((void**)&buffer, image.width * image.height * 3 * sizeof(float)));
  RGBToTensor((const unsigned char*)image.buffer, buffer, image.width, image.height);
  CheckCudaError();

  const int input_tensor_size = 2;

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"logits", "pred_boxes"};



  // https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/
/*
  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                            input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors =
      session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++) {
    std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
  }
  std::cout << std::flush;
  */
  std::cout << "post infer" << std::endl;
  CUDA_SAFE_CALL_NO_SYNC(cudaFree(buffer));
}

resnet_demo::resnet_demo(std::optional<std::string> name_override) : unit::resnet_demo::Base(name_override) {

  inference = std::make_unique<Inference>("/basis_test_robot/docker/model.onnx");

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
