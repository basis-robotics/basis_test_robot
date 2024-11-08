/*

  DO NOT EDIT THIS FILE

  This is a template for use with your Unit, to use as a base, provided as an example.

*/
#include <unit/rpi_libcamera_driver/unit_base.h>

class rpi_libcamera_driver : public unit::rpi_libcamera_driver::Base {
public:
  rpi_libcamera_driver(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::rpi_libcamera_driver::Base(args, name_override)
  {}


  virtual unit::rpi_libcamera_driver::OnCameraImage::Output
  OnCameraImage(const unit::rpi_libcamera_driver::OnCameraImage::Input &input) override;

};