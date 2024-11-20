/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/rpi_freenove_servo_driver/unit_base.h>

#include <PiPCA9685/PCA9685.h>

class rpi_freenove_servo_driver : public unit::rpi_freenove_servo_driver::Base {
public:
  rpi_freenove_servo_driver(const Args& args, const std::optional<std::string_view>& name_override = {});

  virtual unit::rpi_freenove_servo_driver::Update::Output
  Update(const unit::rpi_freenove_servo_driver::Update::Input &input) override;

  virtual unit::rpi_freenove_servo_driver::RequestState0::Output
  RequestState0(const unit::rpi_freenove_servo_driver::RequestState0::Input &input) override;

  virtual unit::rpi_freenove_servo_driver::RequestState1::Output
  RequestState1(const unit::rpi_freenove_servo_driver::RequestState1::Input &input) override;

  virtual unit::rpi_freenove_servo_driver::OnInputs::Output
  OnInputs(const unit::rpi_freenove_servo_driver::OnInputs::Input &input) override;


  PiPCA9685::PCA9685 pca;

  static inline constexpr size_t NUM_SERVOS = 2;

  std::array<double, NUM_SERVOS> current_state;
  std::array<double, NUM_SERVOS> requested_state;

  std::shared_ptr<const basis::robot::input::InputState> last_input;
};