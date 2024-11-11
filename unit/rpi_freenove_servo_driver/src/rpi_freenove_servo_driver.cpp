/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <rpi_freenove_servo_driver.h>

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
  
  // while(true) {
  //   pca.set_pwm(0, 0, 370);
  //   usleep(1'000'000);
  //   pca.set_pwm(0, 0, 415);
  //   usleep(1'000'000);
  //   pca.set_pwm(0, 0, 460);
  //   usleep(1'000'000);
  //   pca.set_pwm(0, 0, 415);
  //   usleep(1'000'000);
  // }
}

Update::Output rpi_freenove_servo_driver::Update(const Update::Input& input) {
  const float t = basis::core::MonotonicTime::Now().ToSeconds();
  //BASIS_LOG_INFO("Requested angle: {}", R2D(sin(t / 20.0))) * (70.0 / 90.0);

  requested_state[0] = (sin(t * 2.0)) * 70.0;
  requested_state[1] = (sin(t * 3.1)) * 60.0;

  std::array<std::shared_ptr<google::protobuf::DoubleValue>, NUM_SERVOS> outputs;
  for(int i = 0; i < NUM_SERVOS; i++) {
    // TODO: smoothing
    current_state[i] = requested_state[i];
    outputs[i] = std::make_shared<google::protobuf::DoubleValue>();
    outputs[i]->set_value(current_state[i]);
    float ms = DegressToPWMMS(current_state[i]);
 //   BASIS_LOG_INFO("Setting servo {} to {}deg {}ms", i, current_state[i], ms);
  
    pca.set_pwm_ms(8 + i, ms);
  }


  // Magic - convert from our array output to our output type
  return std::apply([](auto&&... args) { return Update::Output{args...}; }, std::tuple_cat(outputs));
}

RequestState0::Output rpi_freenove_servo_driver::RequestState0(const RequestState0::Input& input) {
  requested_state[0] = input.servo_0_request_degrees->value();
  return {};
}

RequestState1::Output rpi_freenove_servo_driver::RequestState1(const RequestState1::Input& input) {
  requested_state[1] = input.servo_1_request_degrees->value();
  return {};
}
