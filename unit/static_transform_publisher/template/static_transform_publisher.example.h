/*

  DO NOT EDIT THIS FILE

  This is a template for use with your Unit, to use as a base, provided as an example.

*/
#include <unit/static_transform_publisher/unit_base.h>

class static_transform_publisher : public unit::static_transform_publisher::Base {
public:
  static_transform_publisher(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::static_transform_publisher::Base(args, name_override)
  {}


  virtual unit::static_transform_publisher::Publish::Output
  Publish(const unit::static_transform_publisher::Publish::Input &input) override;

};