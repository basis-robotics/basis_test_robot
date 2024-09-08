/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/resnet_demo/unit_base.h>
#include <onnxruntime_cxx_api.h>


class Inference {
public:
  Inference(const char* model_path);

  void Infer(const image_conversion::CudaManagedImage& image);

  void DetectInputsOutputs();
private:
  std::unique_ptr<Ort::Session> session;
};

class resnet_demo : public unit::resnet_demo::Base {
public:
  resnet_demo(std::optional<std::string> name_override = {});


  virtual unit::resnet_demo::OnRGB::Output
  OnRGB(const unit::resnet_demo::OnRGB::Input &input) override;

  std::unique_ptr<Inference> inference;
};