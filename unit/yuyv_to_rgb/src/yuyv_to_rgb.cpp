/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <yuyv_to_rgb.h>

using namespace unit::yuyv_to_rgb;


OnYUYV::Output yuyv_to_rgb::OnYUYV(const OnYUYV::Input& input) {

  auto rgb = YUYV_to_RGB(*input.camera_yuyv_cuda.get());
  auto foxglove = rgb->ToFoxglove();
  return {std::move(rgb), foxglove};
}
