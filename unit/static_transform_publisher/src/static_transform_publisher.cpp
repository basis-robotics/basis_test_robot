/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <static_transform_publisher.h>
#include <tf2_basis/tf2_basis.h>
#include <google/protobuf/util/time_util.h>

using namespace unit::static_transform_publisher;


Publish::Output static_transform_publisher::Publish(const Publish::Input& input) {
  auto transforms = std::make_shared<foxglove::FrameTransforms>();
  
  if constexpr(std::string_view( "BASIS_PLATFORM" ).compare("ORIN"))
  {
    auto transform = transforms->add_transforms();
    *transform->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(basis::core::MonotonicTime::Now().nsecs);
    transform->mutable_translation()->set_z(0.6);

    tf2::Quaternion rotation;
    rotation.setRPY(0, 0.45, 0);
    transform->mutable_rotation()->set_x(rotation.getX());
    transform->mutable_rotation()->set_y(rotation.getY());
    transform->mutable_rotation()->set_z(rotation.getZ());
    transform->mutable_rotation()->set_w(rotation.getW());
    transform->set_parent_frame_id("robot");
    transform->set_child_frame_id("webcam");
  }

  return {transforms};
}
