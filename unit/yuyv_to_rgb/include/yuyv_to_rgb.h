/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/yuyv_to_rgb/unit_base.h>

class yuyv_to_rgb : public unit::yuyv_to_rgb::Base {
public:
  yuyv_to_rgb(const unit::yuyv_to_rgb::Args &args, const std::optional<std::string_view> &name_override = {})
      : unit::yuyv_to_rgb::Base(args, name_override) {}

  virtual unit::yuyv_to_rgb::OnYUYV::Output OnYUYV(const unit::yuyv_to_rgb::OnYUYV::Input &input) override;
};