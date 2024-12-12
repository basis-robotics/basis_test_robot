/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <rpi_freenove_servo_driver.h>
#include <google/protobuf/util/time_util.h>
#include <tf2_basis/tf2_basis.h>

using namespace unit::rpi_freenove_servo_driver;

float D2R(float degrees) {
  return ( degrees * M_PI ) / 180.0 ;
}

float R2D(float radians) {
  return ( radians * 180.0 ) / M_PI;
}

float DegressToPWMMS(float degrees) {
  constexpr float ms_0 = 1.5;
  constexpr float ms_90 = 0.5;

  constexpr float LIMIT = 70;

  if(degrees < -LIMIT){ 
    degrees = -LIMIT;
  }
  if(degrees > LIMIT) {
    degrees = LIMIT;
  }
  
  return /* the midpoint */ ms_0 + (degrees) /*degrees centered on the midpoint*/ * (ms_90 / 90.0) /* scaled */;
}

rpi_freenove_servo_driver::rpi_freenove_servo_driver(const Args& args, const std::optional<std::string_view>& name_override) 
  : unit::rpi_freenove_servo_driver::Base(args, name_override), pca(args.i2c_device, args.address), current_state {args.default_angle_0, args.default_angle_1}, requested_state {args.default_angle_0, args.default_angle_1}
{
  pca.set_pwm_freq(50.0);
}

Update::Output rpi_freenove_servo_driver::Update(const Update::Input& input) {
  const auto now =  basis::core::MonotonicTime::Now();
  const float t = now.ToSeconds();
  if(input.servo_0_request_degrees) {
    requested_state[0] = input.servo_0_request_degrees->value();
  }
  if(input.servo_1_request_degrees) {
    requested_state[1] = input.servo_1_request_degrees->value();
  }

  if(input.user_inputs && !input.user_inputs->joysticks().empty()) {
    // If we have a joystick connected, use it
    constexpr float MAX_JOYSTICK_DEGREES_SEC = 180.0f;
    // Get the update rate for this handler
    const auto duration = handlers["Update"]->rate_duration; 

    // TODO: move to config
    constexpr size_t AXIS_IDXES[2] = {2, 5};

    const auto& joystick = input.user_inputs->joysticks()[0];
    for(int i = 0; i < NUM_SERVOS; i++) {
      const float delta = joystick.axes()[AXIS_IDXES[i]] * MAX_JOYSTICK_DEGREES_SEC * duration->ToSeconds();
      requested_state[i] = std::clamp(requested_state[i] - delta, -70.0, 70.0);
    }
  }
  else {
    // Otherwise, rotate back and forth
    // TODO: this logic will get moved out to a separate unit
    requested_state[0] = (sin(t * 2.0)) * 70.0;
    requested_state[1] = (sin(t * 3.1)) * 60.0;
  }

  std::array<std::shared_ptr<google::protobuf::DoubleValue>, NUM_SERVOS> outputs;
  for(int i = 0; i < NUM_SERVOS; i++) {
    // TODO: smoothing might be useful here
    current_state[i] = requested_state[i];
    outputs[i] = std::make_shared<google::protobuf::DoubleValue>();
    outputs[i]->set_value(current_state[i]);
    float ms = DegressToPWMMS(current_state[i]);
  
    pca.set_pwm_ms(8 + i, ms);
  }

  auto transforms = std::make_shared<foxglove::FrameTransforms>();
  {
    auto servo_yaw_to_servo_pitch = transforms->add_transforms();
    {
      *servo_yaw_to_servo_pitch->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now.nsecs);
      servo_yaw_to_servo_pitch->set_parent_frame_id("servo_yaw");
      servo_yaw_to_servo_pitch->set_child_frame_id("servo_pitch");
      servo_yaw_to_servo_pitch->mutable_translation()->set_z(0.04);
      // No changes to the pitch for now, we've broken the servo
    }

    auto robot_to_servo_yaw = transforms->add_transforms();
    {
      *robot_to_servo_yaw->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now.nsecs);
      robot_to_servo_yaw->set_parent_frame_id("robot");
      robot_to_servo_yaw->set_child_frame_id("servo_yaw");
      robot_to_servo_yaw->mutable_translation()->set_y(0.095);
      tf2::Quaternion rotation;
      rotation.setRPY(0, 0, D2R(current_state[0]));
      robot_to_servo_yaw->mutable_rotation()->set_x(rotation.getX());
      robot_to_servo_yaw->mutable_rotation()->set_y(rotation.getY());
      robot_to_servo_yaw->mutable_rotation()->set_z(rotation.getZ());
      robot_to_servo_yaw->mutable_rotation()->set_w(rotation.getW());
    }
  }

  // Magic - convert from our array output to our output type
  return std::apply([&](auto&&... args) { return Update::Output{args..., transforms}; }, std::tuple_cat(outputs));
}
