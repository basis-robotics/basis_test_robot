/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include "basis/core/time.h"
#include "spdlog/spdlog.h"
#include <foxglove/ImageAnnotations.pb.h>
#include <foxglove/Point2.pb.h>
#include <foxglove/PointsAnnotation.pb.h>
#include <foxglove/RawImage.pb.h>
#include <memory>
#include <onnxruntime_c_api.h>
#include <resnet_demo.h>
using namespace unit::resnet_demo;

resnet_demo::resnet_demo(std::optional<std::string> name_override) : unit::resnet_demo::Base(name_override) {
  inference =
      // std::make_unique<InferenceDetrResnet>("/basis_test_robot/models/detr-resnet-50", "orin",
      // InferenceRunMode::INFERENCE);
      std::make_unique<InferenceYoloV9>("/basis_test_robot/models/yolov8n", "orin", InferenceRunMode::INFERENCE);
}
OnRGB::Output resnet_demo::OnRGB(const OnRGB::Input &input) {
  auto latency = basis::core::MonotonicTime::Now() - input.camera_rgb_cuda->time;
  SPDLOG_DEBUG("Latency is {}", std::to_string(latency.ToSeconds()));

  auto before = basis::core::MonotonicTime::Now();
  auto detections = inference->Infer(*input.camera_rgb_cuda.get());
  auto after = basis::core::MonotonicTime::Now();

  SPDLOG_DEBUG("Inference took {}", (after - before).ToSeconds());
  // auto [r, g, b] = inference->DumpInferenceBuffer();
  // return {r, g, b};
  auto annotations_msg = std::make_shared<foxglove::ImageAnnotations>();
  const auto ts = input.camera_rgb_cuda->time.ToTimespec();
  auto points_msgs = annotations_msg->mutable_points();
  points_msgs->Reserve(detections.size());
  auto text_msgs = annotations_msg->mutable_points();
  text_msgs->Reserve(detections.size());
  for (auto &detection : detections) {
    auto points_msg = annotations_msg->add_points();

    points_msg->mutable_timestamp()->set_seconds(ts.tv_sec);
    points_msg->mutable_timestamp()->set_nanos(ts.tv_nsec);
    points_msg->set_type(::foxglove::PointsAnnotation_Type::PointsAnnotation_Type_LINE_LOOP);
    points_msg->set_thickness(5.0);
    points_msg->mutable_outline_color()->set_r(1.0);
    points_msg->mutable_outline_color()->set_a(1.0);
    points_msg->mutable_points()->Reserve(4);
    for (int i = 0; i < 4; i++) {
      auto point_msg = points_msg->add_points();
      point_msg->set_x(detection.bbox[i][0]);
      point_msg->set_y(detection.bbox[i][1]);
    }

    auto text_msg = annotations_msg->add_texts();
    text_msg->mutable_timestamp()->set_seconds(ts.tv_sec);
    text_msg->mutable_timestamp()->set_nanos(ts.tv_nsec);
    text_msg->set_text(detection.det_class);
    text_msg->mutable_position()->set_x(detection.bbox[0][0]);
    text_msg->mutable_position()->set_y(detection.bbox[0][1]);
    text_msg->set_font_size(30.0);
    text_msg->mutable_text_color()->set_r(1.0);
    text_msg->mutable_text_color()->set_a(1.0);
  }

  auto [r, g, b] = inference->DumpInferenceBuffer();
  return {annotations_msg, r, g, b};
  // return {annotations_msg};
}
