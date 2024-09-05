#include <v4l2_camera_driver.h>

using namespace unit::v4l2_camera_driver;

using namespace basis;
using namespace basis::core;

void v4l2_camera_driver::CameraUpdateLoop() {
  const uint64_t run_token = MonotonicTime::GetRunToken();

  auto sleep_until = MonotonicTime::Now();
  while(!stop && MonotonicTime::GetRunToken() == run_token) {
    BASIS_LOG_INFO("Tick");
    sleep_until += Duration::FromSeconds(1);
    sleep_until.SleepUntil(run_token);
  }
}

OnCameraImage::Output v4l2_camera_driver::OnCameraImage(const OnCameraImage::Input& input) {
  OnCameraImage::Output output;

  return output;
}
