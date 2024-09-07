/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <resnet_demo.h>

#include <onnxruntime_cxx_api.h>
using namespace unit::resnet_demo;



Inference::Inference(std::string_view model) 
//:
//  ort_env(ORT_LOGGING_LEVEL_WARNING, "inference")
{
  const auto& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* tensorrt_options;

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  const char* model_path = "/basis_test_robot/docker/model.onnx";

  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
      tensorrt_options, api.ReleaseTensorRTProviderOptions);
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options),
                                                        rel_trt_options.get()));

  std::cout << "Running ORT TRT EP with default provider options" << std::endl;
}


resnet_demo::resnet_demo(std::optional<std::string> name_override)
  : unit::resnet_demo::Base(name_override)
  {


  //Ort::Session session(env, model_path, session_options);
#if 0
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

   // print number of model input nodes
  const size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_dims;

  std::cout << "Number of inputs = " << num_input_nodes << std::endl;

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

    // print input node types
    auto type_info = session.GetInputTypeInfo(i);
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
OnRGB::Output resnet_demo::OnRGB(const OnRGB::Input& input) {
  return {};
}
