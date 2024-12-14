/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/rplidar_driver/unit_base.h>

#include "sl_lidar.h" 
#include "sl_lidar_driver.h"


class rplidar_driver : public unit::rplidar_driver::Base {
public:
  rplidar_driver(const Args& args, const std::optional<std::string_view>& name_override = {}) 
    : args(args), unit::rplidar_driver::Base(args, name_override)
  {}

  ~rplidar_driver();

  virtual unit::rplidar_driver::OnLidar::Output
  OnLidar(const unit::rplidar_driver::OnLidar::Input &input) override;

  Args args;

  std::unique_ptr<sl::IChannel> channel;
  std::unique_ptr<sl::ILidarDriver> lidar;
  sl::LidarScanMode scan_mode = {};

  sl_lidar_response_device_info_t device_info = {};
};