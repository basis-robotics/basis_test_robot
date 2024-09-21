/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include "basis/core/logging/macros.h"
#include "basis/core/transport/convertable_inproc.h"
#include <memory>
#include <yuyv_to_rgb.h>

using namespace unit::yuyv_to_rgb;

OnYUYV::Output yuyv_to_rgb::OnYUYV(const OnYUYV::Input &input) {
  std::shared_ptr<const image_conversion::CudaManagedImage> yuyv;
  switch (input.camera_yuyv.index()) {
  case basis::MessageVariant::NO_MESSAGE:
    BASIS_LOG_ERROR("Got empty variant");
    return {};
  case basis::MessageVariant::DESERIALIZED_MESSAGE:
    BASIS_LOG_ERROR("got type");
    // todo:make nicer helper for this
    return {};
  case basis::MessageVariant::INPROC_MESSAGE:
    BASIS_LOG_ERROR("got inproc type");
    yuyv = std::get<2>(input.camera_yuyv);
    break;
  }

  std::shared_ptr<image_conversion::CudaManagedImage> rgb = YUYV_to_RGB(*yuyv);

  return {rgb};
}
