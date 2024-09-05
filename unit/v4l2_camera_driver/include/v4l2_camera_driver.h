#pragma once
#include <linux/videodev2.h>
#include <unit/v4l2_camera_driver/unit_base.h>
class v4l2_camera_driver : public unit::v4l2_camera_driver::Base {
public:
  v4l2_camera_driver(std::optional<std::string> name_override = {}) : unit::v4l2_camera_driver::Base(name_override) {}

  ~v4l2_camera_driver();

  bool InitializeCamera(std::string_view camera_device);
  void CloseCamera();

  bool Queue(int index);
  bool Dequeue(int index);

  virtual unit::v4l2_camera_driver::OnCameraImage::Output
  OnCameraImage(const unit::v4l2_camera_driver::OnCameraImage::Input &input) override;
  int camera_fd = -1;
  static constexpr int BUFFER_COUNT = 2;
  // TODO: this is technically leaked
  char *camera_buffers[BUFFER_COUNT] = {};
  v4l2_buffer buffer_infos[BUFFER_COUNT] = {};

  v4l2_format imageFormat;
  int current_index = 0;
};