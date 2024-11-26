/*

*/

#include <rpi_freenove_mecanum_driver.h>

#include <google/protobuf/util/time_util.h>

using namespace unit::rpi_freenove_mecanum_driver;

std::array<float, 4> XYTtoWheels(float x, float y, float theta) {
  constexpr float MAX_SPEED = 10.0f; // TODO: units
  // simple enough?
  // https://robotics.stackexchange.com/questions/20088/how-to-drive-mecanum-wheels-robot-code-or-algorithm
  // This isn't quite what we want, I think the /3.0 is bad
  return {
    -MAX_SPEED * (y+x-theta),
    MAX_SPEED * (y-x-theta), // Note: the hardware for this device has one reversed motor
    -MAX_SPEED * (y+x+theta),
    -MAX_SPEED * (y-x+theta),
  };
}

rpi_freenove_mecanum_driver::rpi_freenove_mecanum_driver(const Args& args, const std::optional<std::string_view>& name_override)
  : unit::rpi_freenove_mecanum_driver::Base(args, name_override), pca(args.i2c_device, args.address)
{
  pca.set_pwm_freq(50.0);
}

Update::Output rpi_freenove_mecanum_driver::Update(const Update::Input& input) {
  auto motor_state = std::make_shared<basis::robot::state::MotorState>();

  *motor_state->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(basis::core::MonotonicTime::Now().nsecs);

  for(int i = 0; i < basis::robot::state::MotorIndexes_ARRAYSIZE; i++) {
    motor_state->mutable_motors()->Add();
  }

  if(input.user_inputs && input.user_inputs->joysticks().size()) {
    auto& joystick = input.user_inputs->joysticks()[0];
    constexpr int X_AXIS_IDX = 0;
    constexpr int Y_AXIS_IDX = 1;
    constexpr int TURN_LEFT_IDX = 9;
    constexpr int TURN_RIGHT_IDX = 10;

    float x = joystick.axes()[X_AXIS_IDX];
    float y = -joystick.axes()[Y_AXIS_IDX];
    float theta = ((1 + joystick.axes()[TURN_RIGHT_IDX]) - (1 + joystick.axes()[TURN_LEFT_IDX])) / 2.0f;

    auto* command = motor_state->mutable_command();

    command->set_x(x);
    command->set_y(y);
    command->set_theta(theta);

    auto commands = XYTtoWheels(x, y, theta);
    for(int i = 0; i < basis::robot::state::MotorIndexes_ARRAYSIZE; i++) {
      motor_state->mutable_motors()->Mutable(i)->set_commanded_speed(commands[i]);

      float pos = 0.0;
      float neg = 0.0;
      if(commands[i] > 1.0) {
        pos = commands[i];
      }
      else if (commands[i] < -1.0) {
        neg = -commands[i];
      }
      pca.set_pwm_ms(2*i, pos);
      pca.set_pwm_ms(2*i + 1, neg);
    }

    

  }

  return {
    std::move(motor_state)
  };
}