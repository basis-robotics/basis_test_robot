/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/perception_demo/unit_base.h>
#include <onnxruntime_cxx_api.h>

#include "inference.h"

class perception_demo : public unit::perception_demo::Base {
public:
  perception_demo(const unit::perception_demo::Args& args, const std::optional<std::string_view>& name_override = {});

  virtual unit::perception_demo::OnRGB::Output
  OnRGB(const unit::perception_demo::OnRGB::Input &input) override;

  std::unique_ptr<Inference> inference;
  unit::perception_demo::Args args;
};