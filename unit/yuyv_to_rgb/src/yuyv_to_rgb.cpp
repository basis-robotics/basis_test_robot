/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include "basis/core/logging/macros.h"
#include "basis/core/transport/convertable_inproc.h"
#include "image_conversion.h"
#include <memory>
#include <yuyv_to_rgb.h>

using namespace unit::yuyv_to_rgb;

OnYUYV::Output yuyv_to_rgb::OnYUYV(const OnYUYV::Input &input) {
  std::shared_ptr<const image_conversion::CudaManagedImage> yuyv =
      image_conversion::CudaManagedImage::FromVariant(input.args_topic_namespace_yuyv);
  if(!yuyv) {
    return {};
  }
  std::shared_ptr<image_conversion::CudaManagedImage> rgb = image_conversion::YUYV_to_RGB(*yuyv);

  return {rgb};
}
