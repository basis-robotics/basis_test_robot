/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/resnet_demo/unit_base.h>
#include <onnxruntime_cxx_api.h>

#include "inference.h"

class resnet_demo : public unit::resnet_demo::Base {
public:
  resnet_demo(const unit::resnet_demo::Args& args, const std::optional<std::string_view>& name_override = {});

  virtual unit::resnet_demo::OnRGB::Output
  OnRGB(const unit::resnet_demo::OnRGB::Input &input) override;

  std::unique_ptr<Inference> inference;
  unit::resnet_demo::Args args;
};