#include <unit/v4l2_camera_driver/unit_base.h>

class v4l2_camera_driver : public unit::v4l2_camera_driver::Base {
public:
  v4l2_camera_driver(std::optional<std::string> name_override = {}) 
  : unit::v4l2_camera_driver::Base(name_override)
  {
    BASIS_LOG_INFO("starting thread");
    camera_thread = std::thread([this](){
      CameraUpdateLoop();
    });
  }

  ~v4l2_camera_driver() {
    stop = true;

    camera_thread.join();
  }

  void CameraUpdateLoop();

  virtual unit::v4l2_camera_driver::OnCameraImage::Output
  OnCameraImage(const unit::v4l2_camera_driver::OnCameraImage::Input &input) override;

  std::thread camera_thread;
  std::atomic<bool> stop = false;
};