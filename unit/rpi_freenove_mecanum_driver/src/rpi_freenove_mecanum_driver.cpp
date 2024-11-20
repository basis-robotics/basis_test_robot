/*

*/

#include <rpi_freenove_mecanum_driver.h>

using namespace unit::rpi_freenove_mecanum_driver;

Update::Output rpi_freenove_mecanum_driver::Update(const Update::Input& input) {
  return {};
}

OnInputs::Output rpi_freenove_mecanum_driver::OnInputs(const OnInputs::Input& input) {
  last_input = input.user_inputs;

  return {};
}
