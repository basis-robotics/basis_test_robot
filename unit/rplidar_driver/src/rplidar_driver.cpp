/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/

#include <rplidar_driver.h>

#include <google/protobuf/util/time_util.h>

using namespace unit::rplidar_driver;

const char* SlErrorToString(sl_result error) {
  switch(error) {
    case SL_RESULT_OK: return "SL_RESULT_OK";
    case SL_RESULT_ALREADY_DONE: return "SL_RESULT_ALREADY_DONE";
    case SL_RESULT_INVALID_DATA: return "SL_RESULT_INVALID_DATA";
    case SL_RESULT_OPERATION_FAIL: return "SL_RESULT_OPERATION_FAIL";
    case SL_RESULT_OPERATION_TIMEOUT: return "SL_RESULT_OPERATION_TIMEOUT";
    case SL_RESULT_OPERATION_STOP: return "SL_RESULT_OPERATION_STOP";
    case SL_RESULT_OPERATION_NOT_SUPPORT: return "SL_RESULT_OPERATION_NOT_SUPPORT";
    case SL_RESULT_FORMAT_NOT_SUPPORT: return "SL_RESULT_FORMAT_NOT_SUPPORT";
    case SL_RESULT_INSUFFICIENT_MEMORY: return "SL_RESULT_INSUFFICIENT_MEMORY";
  }
  return "UNKNOWN";
}

float D2R(float degrees) {
  return ( degrees * M_PI ) / 180.0 ;
}

float getAngle(uint16_t angle_z_q14) {
  return D2R(angle_z_q14 * 90.f / 16384.f);
}

OnLidar::Output rplidar_driver::OnLidar(const OnLidar::Input& input) {
  if(!channel) {
    sl::Result<sl::IChannel*> maybe_channel = sl::createSerialPortChannel(args.device, 460800);

    if(maybe_channel) {
      channel = std::unique_ptr<sl::IChannel>(*maybe_channel);
    }
    else {
      BASIS_LOG_ERROR("Failed to open serial port channel: {}", SlErrorToString(maybe_channel.err));
      return {};
    }
  }

  if(lidar && !lidar->isConnected()) {
    BASIS_LOG_ERROR("LIDAR disconnected, reconnecting...");
    lidar = nullptr;
  }

  if(!lidar) {
    sl::Result<sl::ILidarDriver*> maybe_lidar_driver = *sl::createLidarDriver();
    if(!maybe_lidar_driver) {
      BASIS_LOG_ERROR("Failed to create LIDAR driver: {}", SlErrorToString(maybe_lidar_driver.err));
      return {};
    }
    sl_result res = (*maybe_lidar_driver)->connect(channel.get());
    if(SL_IS_FAIL(res)) { 
      BASIS_LOG_ERROR("Failed to connect LIDAR driver: {}", SlErrorToString(res));
      return {};
    }

    if((*maybe_lidar_driver)->isConnected()) {
      BASIS_LOG_INFO("Connected");
    }

    res = (*maybe_lidar_driver)->getDeviceInfo(device_info);
    if(SL_IS_FAIL(res)){
      BASIS_LOG_ERROR("Failed to get device information from LIDAR (is your baud rate set correctly?): {}", SlErrorToString(res));    
      return {};
    }
    else {
      BASIS_LOG_INFO("Model: {}, Firmware Version: {}.{}, Hardware Version: {}",
          device_info.model,
          device_info.firmware_version >> 8, device_info.firmware_version & 0xffu,
          device_info.hardware_version);
    }

    sl_lidar_response_device_health_t healthinfo;

    const sl_result op_result = (*maybe_lidar_driver)->getHealth(healthinfo);
    if (SL_IS_OK(op_result)) { 
        //BASIS_LOG_INFO("RPLidar health status : %d", healthinfo.status);
      switch (healthinfo.status) {
			  case SL_LIDAR_STATUS_OK:
          BASIS_LOG_INFO("RPLidar health status : OK.");
        break;
			case SL_LIDAR_STATUS_WARNING:
        BASIS_LOG_INFO("RPLidar health status : Warning.");
        break;
			case SL_LIDAR_STATUS_ERROR:
        BASIS_LOG_ERROR("Error, rplidar internal error detected. Please reboot the device to retry.");
        break;
      default:
        BASIS_LOG_ERROR("Error, Unknown internal error detected. Please reboot the device to retry.");
        break;
      }
    } else {
      BASIS_LOG_ERROR("Error, cannot retrieve rplidar health code: {}", SlErrorToString(op_result));
    }
    
    std::vector<sl::LidarScanMode> scan_modes;
    sl_result scan_mode_res = (*maybe_lidar_driver)->getAllSupportedScanModes(scan_modes);
    if(SL_IS_FAIL(scan_mode_res)){
      BASIS_LOG_ERROR("Failed to get supported scan modes: {}", SlErrorToString(scan_mode_res));
      return {};
    }

    BASIS_LOG_INFO("Scan modes:");
    for(auto& mode : scan_modes) {
      BASIS_LOG_INFO("\t{}: {} {} {} {}", mode.id, mode.us_per_sample, mode.max_distance, mode.ans_type, mode.scan_mode);
      if(strcmp(mode.scan_mode, "Standard")) {
        scan_mode = mode;
      }
    }
    if(scan_mode.scan_mode[0] == 0) {
      BASIS_LOG_ERROR("No 'Standard' scan mode");
    }

    lidar = std::unique_ptr<sl::ILidarDriver>(*maybe_lidar_driver);

    // Call once to initialize
    lidar->startScan(false, false, 0, &scan_mode);
  }

  constexpr size_t MAX_NODE_COUNT = 8192;

  size_t node_count = MAX_NODE_COUNT;
  sl_lidar_response_measurement_node_hq_t nodes[MAX_NODE_COUNT];
  sl_u64 timestamp = 0;
  const sl_result scan_data_result = lidar->grabScanDataHq(nodes, node_count, timestamp);
  if(timestamp == 0) {
    // This appears to never be set on the C1
    timestamp = basis::core::MonotonicTime::Now().nsecs;
  }

  if (SL_IS_FAIL(scan_data_result))
  {
      BASIS_LOG_ERROR("Failed to get LIDAR scan: {}", SlErrorToString(scan_data_result));
      return {};
  }
  
  BASIS_LOG_DEBUG("Got scan with ts {} and count {}", timestamp, node_count);

  auto out = std::make_shared<foxglove::LaserScan>();

  *out->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(timestamp);
  out->set_frame_id(args.sensor_namespace.c_str() + 1);
  auto ranges = out->mutable_ranges();
  ranges->Resize(node_count, 0.0);

  auto intensities = out->mutable_intensities();
  intensities->Resize(node_count, 0.0);
  for(int i = 0; i < node_count; i++) {
    const auto* node = nodes + i;
    const float dist = node->dist_mm_q2 / 4.0f / 1000;
    if (dist == 0.0) {
      (*ranges)[i] = std::numeric_limits<float>::infinity();
    }
    else {
      (*ranges)[i] = dist;
    }
    (*intensities)[i] = float(node->quality >> 2);
  }
  if(node_count) {
    float min = getAngle(nodes[0].angle_z_q14);
    float max = getAngle(nodes[node_count - 1].angle_z_q14);
    
    if(min > max) {
      std::swap(min, max);
    }
    
    // Usually min and max aren't straddling 0 radians - fix this
    if(max - min < M_PI)
    {
      min -= 2.0 * M_PI;
    }

    // Put the angles into the rotation space we expect
    out->set_start_angle(M_PI - min);
    out->set_end_angle(M_PI - max);
  }
  return {out};
}

rplidar_driver::~rplidar_driver() {
  if(lidar) {
    lidar->stop();
  }
}