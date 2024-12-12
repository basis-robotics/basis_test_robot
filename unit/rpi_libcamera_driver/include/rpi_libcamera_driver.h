/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/rpi_libcamera_driver/unit_base.h>

#include <libcamera/libcamera.h>

#include <image_conversion.h>
#include <mutex>

class rpi_libcamera_driver : public unit::rpi_libcamera_driver::Base {
public:
  rpi_libcamera_driver(const Args& args, const std::optional<std::string_view>& name_override = {});
  ~rpi_libcamera_driver() {
    if(camera) {
      camera->stop();
      camera->release();
    }
    camera = nullptr;
    camera_manager->stop();
  }

  virtual unit::rpi_libcamera_driver::OnCameraImage::Output
  OnCameraImage(const unit::rpi_libcamera_driver::OnCameraImage::Input &input) override;

protected:
  void OnRequestComplete(libcamera::Request* request);

  const Args args;

  std::unique_ptr<libcamera::CameraManager> camera_manager;
  std::shared_ptr<libcamera::Camera> camera;
  std::unique_ptr<libcamera::FrameBufferAllocator> frame_allocator;
  std::vector<std::unique_ptr<libcamera::Request>> requests;

  
  std::shared_ptr<image_conversion::Image> next_output;
  std::mutex next_output_lock;
};