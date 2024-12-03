# TODO: rename to freenove_servos

FetchContent_Declare(
  PiPCA9685 
  GIT_REPOSITORY https://github.com/basis-robotics/PiPCA9685.git
  GIT_TAG main
)
FetchContent_MakeAvailable(PiPCA9685)

generate_unit(rpi_freenove_servo_driver)
    
target_link_libraries(unit_rpi_freenove_servo_driver tf2_basis PiPCA9685_cpp)
