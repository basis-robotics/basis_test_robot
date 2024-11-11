/*

  DO NOT EDIT THIS FILE

  This is a template for use with your Unit, to use as a base, provided as an example.

*/
#include <unit/rpi_freenove_servo_driver/unit_base.h>

class rpi_freenove_servo_driver : public unit::rpi_freenove_servo_driver::Base {
public:
  rpi_freenove_servo_driver(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::rpi_freenove_servo_driver::Base(args, name_override)
  {}


  virtual unit::rpi_freenove_servo_driver::Update::Output
  Update(const unit::rpi_freenove_servo_driver::Update::Input &input) override;

  virtual unit::rpi_freenove_servo_driver::RequestState0::Output
  RequestState0(const unit::rpi_freenove_servo_driver::RequestState0::Input &input) override;

  virtual unit::rpi_freenove_servo_driver::RequestState1::Output
  RequestState1(const unit::rpi_freenove_servo_driver::RequestState1::Input &input) override;

};