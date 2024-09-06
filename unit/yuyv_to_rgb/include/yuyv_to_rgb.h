/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/yuyv_to_rgb/unit_base.h>

class yuyv_to_rgb : public unit::yuyv_to_rgb::Base {
public:
  yuyv_to_rgb(std::optional<std::string> name_override = {}) 
  : unit::yuyv_to_rgb::Base(name_override)
  {}


  virtual unit::yuyv_to_rgb::OnYUYV::Output
  OnYUYV(const unit::yuyv_to_rgb::OnYUYV::Input &input) override;

};