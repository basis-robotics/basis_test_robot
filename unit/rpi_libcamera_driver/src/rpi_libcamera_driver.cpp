/*

adopted from https://www.libcamera.org/guides/application-developer.html

*/

#include <rpi_libcamera_driver.h>

#include <libcamera/formats.h>

#include <sys/mman.h>
#include <unistd.h>

using namespace unit::rpi_libcamera_driver;

// TODO: bind libcamera logger

rpi_libcamera_driver::rpi_libcamera_driver(const Args& args, const std::optional<std::string_view>& name_override) 
  : args(args), unit::rpi_libcamera_driver::Base(args, name_override)
  {
    camera_manager = std::make_unique<libcamera::CameraManager>();
    camera_manager->start();
  }

OnCameraImage::Output rpi_libcamera_driver::OnCameraImage(const OnCameraImage::Input& input) {
  if(!camera) {
    for (auto const &active_camera : camera_manager->cameras())
    {
      if(!args.device || active_camera->id() == *args.device) {
        BASIS_LOG_INFO("Selecting camera {}", active_camera->id());
      }
      active_camera->acquire();
      camera = active_camera;
    }
    if(!camera) {
      BASIS_LOG_WARN("No suitable camera found.");
      return {};
    }
    std::unique_ptr<libcamera::CameraConfiguration> config = camera->generateConfiguration( { libcamera::StreamRole::VideoRecording} );

    libcamera::StreamConfiguration &stream_config = config->at(0);
    BASIS_LOG_INFO("Default configuration is: {}", stream_config.toString());
    
    // note: R and B get swapped, somehow...
    stream_config.pixelFormat = libcamera::formats::BGR888;
    if(args.width) {
      stream_config.size.width = args.width;
    }
    if(args.height) {
      stream_config.size.height = args.height;
    }
    config->validate();
    BASIS_LOG_INFO("Validated configuration is: {}", stream_config.toString());

    if(args.width && args.width != stream_config.size.width)
    {
      BASIS_LOG_ERROR("Couldn't set width to {}, got back {} instead", args.width,stream_config.size.width);
      goto error;
    }
    if(args.height && args.height != stream_config.size.height) {
      BASIS_LOG_ERROR("Couldn't set height to {}, got back {} instead", args.height,stream_config.size.height);
      goto error;
    }
    

    camera->configure(config.get());

    frame_allocator = std::make_unique<libcamera::FrameBufferAllocator>(camera);

    for (libcamera::StreamConfiguration &cfg : *config) {
      int ret = frame_allocator->allocate(cfg.stream());
      if (ret < 0) {
          BASIS_LOG_ERROR("Can't allocate buffers");
          goto error;
      }

      size_t allocated = frame_allocator->buffers(cfg.stream()).size();
      BASIS_LOG_INFO("Allocated {} buffers for stream", allocated);
    }
    
    libcamera::Stream *stream = stream_config.stream();
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& frame_buffers = frame_allocator->buffers(stream);

    for (const std::unique_ptr<libcamera::FrameBuffer> &buffer : frame_buffers) {
      std::unique_ptr<libcamera::Request> request = camera->createRequest();
      if (!request)
      {
          BASIS_LOG_ERROR("Can't create request");
          goto error;
      }

      int ret = request->addBuffer(stream, buffer.get());
      if (ret < 0)
      {
          BASIS_LOG_ERROR("Can't set buffer for request");
          goto error;
      }

      // 1000000/30
      std::int64_t value_pair[2] = {33333,33333};
      request->controls().set(libcamera::controls::FrameDurationLimits, libcamera::Span<const std::int64_t, 2>(value_pair));

      requests.push_back(std::move(request));
    }

    camera->requestCompleted.connect(this, &rpi_libcamera_driver::OnRequestComplete);
    camera->start();
    for (std::unique_ptr<libcamera::Request> &request : requests){
      camera->queueRequest(request.get());
    }

  }

  if(camera)
  {
    std::unique_lock lock(next_output_lock);
    return {std::move(next_output)};
  }
  
error:
  camera->release();
  camera = nullptr;
  return {};
}

void rpi_libcamera_driver::OnRequestComplete(libcamera::Request* request) {
  BASIS_LOG_INFO("rpi_libcamera_driver::OnRequestComplete");
  if (request->status() == libcamera::Request::RequestCancelled){
    return;
  }
  
  const std::map<const libcamera::Stream *, libcamera::FrameBuffer *> &buffers = request->buffers();
  
  assert(buffers.size() == 1);
  for (auto& bufferPair : buffers) {
    libcamera::FrameBuffer *buffer = bufferPair.second;
    const libcamera::FrameMetadata &metadata = buffer->metadata();
    
 
    // TODO: later when we have a better handle on threading in basis, this can publish directly

    // Quick math to convert from system time to monotonic
    // TODO: move to core
    const int64_t capture_systime = metadata.timestamp;
    const int64_t now_monotonic = basis::core::MonotonicTime::Now().nsecs;
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t now_system = std::nano::den * ts.tv_sec + ts.tv_nsec;
    const int64_t capture_monotonic = capture_systime + now_monotonic - now_system;

  
    const auto& plane = buffer->planes()[0];
    const int fd = plane.fd.get();
    const size_t length = lseek(fd, 0, SEEK_END);

    // Note: this could very well live in a class inheriting from image_conversion::Image if we wanted to keep the images in DMA land
    void *address = mmap(nullptr, length, PROT_READ,
              MAP_SHARED, fd, 0);
    if (address == MAP_FAILED) {
      BASIS_LOG_ERROR("Failed to mmap plane: {}", strerror(errno));
      return;
    }

    {
      std::unique_lock lock(next_output_lock);
      next_output = std::make_shared<image_conversion::CpuImage>(image_conversion::PixelFormat::RGB, args.width, args.height, basis::core::MonotonicTime::FromNanoseconds(capture_monotonic), static_cast<std::byte *>(address));
    }
    
  }
  request->reuse(libcamera::Request::ReuseBuffers);
  camera->queueRequest(request);
  
}