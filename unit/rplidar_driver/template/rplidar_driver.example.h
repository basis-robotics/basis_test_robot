/*

  DO NOT EDIT THIS FILE

  This is a template for use with your Unit, to use as a base, provided as an example.

*/
#include <unit/rplidar_driver/unit_base.h>

class rplidar_driver : public unit::rplidar_driver::Base {
public:
  rplidar_driver(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::rplidar_driver::Base(args, name_override)
  {}


  virtual unit::rplidar_driver::OnLidar::Output
  OnLidar(const unit::rplidar_driver::OnLidar::Input &input) override;

};