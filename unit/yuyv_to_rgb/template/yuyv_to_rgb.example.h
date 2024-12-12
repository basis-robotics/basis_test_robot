/*

  DO NOT EDIT THIS FILE

  This is a template for use with your Unit, to use as a base, provided as an example.

*/
#include <unit/yuyv_to_rgb/unit_base.h>

class yuyv_to_rgb : public unit::yuyv_to_rgb::Base {
public:
  yuyv_to_rgb(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::yuyv_to_rgb::Base(args, name_override)
  {}


  virtual unit::yuyv_to_rgb::OnYUYV::Output
  OnYUYV(const unit::yuyv_to_rgb::OnYUYV::Input &input) override;

};