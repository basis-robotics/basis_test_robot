/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/rpi_freenove_mecanum_driver/unit_base.h>

class rpi_freenove_mecanum_driver : public unit::rpi_freenove_mecanum_driver::Base {
public:
  rpi_freenove_mecanum_driver(const Args& args, const std::optional<std::string_view>& name_override = {}) 
  : unit::rpi_freenove_mecanum_driver::Base(args, name_override)
  {}


  virtual unit::rpi_freenove_mecanum_driver::Update::Output
  Update(const unit::rpi_freenove_mecanum_driver::Update::Input &input) override;

  virtual unit::rpi_freenove_mecanum_driver::OnInputs::Output
  OnInputs(const unit::rpi_freenove_mecanum_driver::OnInputs::Input &input) override;

  std::shared_ptr<const basis::robot::input::InputState> last_input;

};