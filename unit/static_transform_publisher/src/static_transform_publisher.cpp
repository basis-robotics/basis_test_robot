/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <static_transform_publisher.h>
#include <tf2_basis/tf2_basis.h>
#include <google/protobuf/util/time_util.h>

using namespace unit::static_transform_publisher;

#define XSTRINGIFY(x) STRINGIFY(x)
#define STRINGIFY(x) #x

static inline constexpr std::string_view PLATFORM = XSTRINGIFY(BASIS_PLATFORM);

Publish::Output static_transform_publisher::Publish(const Publish::Input& input) {
  auto transforms = std::make_shared<foxglove::FrameTransforms>();
  // TODO: maybe we should pass this as an arg instead
  if constexpr(PLATFORM == "ORIN")
  {
    auto transform = transforms->add_transforms();
    *transform->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(basis::core::MonotonicTime::Now().nsecs);
    transform->mutable_translation()->set_z(0.6);
    transform->set_parent_frame_id("robot");
    transform->set_child_frame_id("webcam");
    tf2::Quaternion rotation;
    rotation.setRPY(0, 0.45, 0);
    transform->mutable_rotation()->set_x(rotation.getX());
    transform->mutable_rotation()->set_y(rotation.getY());
    transform->mutable_rotation()->set_z(rotation.getZ());
    transform->mutable_rotation()->set_w(rotation.getW());
  }
  else if constexpr(PLATFORM == "PI")
  {
    auto stamp = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(basis::core::MonotonicTime::Now().nsecs);
    const std::vector static_transforms = {
      // This is dynamic
      //{{0.0, 0.095, 0.0}, {0, 0, 0, 1.0}, , "robot", "servo_yaw"},
      //{{0.0, 0.0, 0.04}, {0, 0, 0, 1.0}, , "servo_yaw", "servo_pitch"},
      tf2_basis::toFoxglove({0.0, 0.01, 0.023}, {0, 0, 0, 1.0}, {}, "servo_pitch", "camera"),
    };
    for(const auto& t : static_transforms) {
      auto transform_msg = transforms->add_transforms();
      transform_msg->CopyFrom(t);
      *transform_msg->mutable_timestamp() = stamp;
    }

  }

  return {transforms};
}
