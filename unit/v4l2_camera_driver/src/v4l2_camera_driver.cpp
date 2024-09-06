#include <fcntl.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <image_conversion.h>

#include <v4l2_camera_driver.h>


// Adapted from
// https://gist.github.com/sammy17/b391c68a91f381aad0d149e325e6a87e
// TODO: translate these
// v4l2-ctl -d /dev/video0 -c exposure_auto=1
// v4l2-ctl -d /dev/video0 -c exposure_absolute=400

using namespace unit::v4l2_camera_driver;

using namespace basis;
using namespace basis::core;

v4l2_camera_driver::~v4l2_camera_driver() { CloseCamera(); }

bool v4l2_camera_driver::InitializeCamera(std::string_view camera_device) {
  if (camera_fd != -1) {
    return true;
  }

  // 1.  Open the device
  int fd = open(std::string(camera_device).c_str(), O_RDWR);
  if (fd < 0) {
    BASIS_LOG_ERROR("Failed to open device, OPEN");
    return false;
  }

  // 2. Ask the device if it can capture frames
  v4l2_capability capability;
  if (ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0) {
    // something went wrong... exit
    BASIS_LOG_ERROR("Failed to get device capabilities, VIDIOC_QUERYCAP");
    close(fd);
    return false;
  }

  // 3. Set Image format
  imageFormat.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  imageFormat.fmt.pix.width = 0;
  imageFormat.fmt.pix.height = 0;
  imageFormat.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  imageFormat.fmt.pix.field = V4L2_FIELD_NONE;


  struct v4l2_fmtdesc fmt;
  struct v4l2_frmsizeenum frmsize;

  int biggest_w = 0;

  char desired_format[] = "YUYV";

  fmt.index = 0;
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while (ioctl(fd, VIDIOC_ENUM_FMT, &fmt) >= 0) {
    // Useful for later, but for now, skip any non YUYV
    if(fmt.pixelformat == V4L2_PIX_FMT_YUYV) {
      frmsize.pixel_format = fmt.pixelformat;
      frmsize.index = 0;
      while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0) {
        int width;
        int height;
        if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
          width = frmsize.discrete.width;
          height = frmsize.discrete.height;
          BASIS_LOG_DEBUG("{}x{}",width, height);

          if(width > imageFormat.fmt.pix.width) {
            imageFormat.fmt.pix.width = width;
            imageFormat.fmt.pix.height = height;
          }
        }
        frmsize.index++;
      }
    }
    fmt.index++;
  }
  
  // Enforce 640x480 as large messages aren't happy yet over TCP
  imageFormat.fmt.pix.width = 640;
  imageFormat.fmt.pix.height = 480;


  if (imageFormat.fmt.pix.width == 0) {
    BASIS_LOG_ERROR("Unable to find a suitable camera format");
    close(fd);
    return false;
  }

  // tell the device you are using this format
  if (ioctl(fd, VIDIOC_S_FMT, &imageFormat) < 0) {
    BASIS_LOG_ERROR("Device could not set format, VIDIOC_S_FMT");
    close(fd);
    return false;
  }
  // Note: this can silently fail...why?
  
  struct v4l2_streamparm streamparm;
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (ioctl(fd, VIDIOC_G_PARM, &streamparm) < 0) {
    BASIS_LOG_ERROR("VIDIOC_G_PARM");
    close(fd);
    return false;
  }

  struct v4l2_frmivalenum frmival;
  memset(&frmival,0,sizeof(frmival));
  frmival.pixel_format = V4L2_PIX_FMT_YUYV;
  frmival.width = imageFormat.fmt.pix.width;
  frmival.height = imageFormat.fmt.pix.height;
  while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) == 0) 
  {
      if (frmival.type == V4L2_FRMIVAL_TYPE_DISCRETE) 
          BASIS_LOG_INFO("{} fps", 1.0*frmival.discrete.denominator/frmival.discrete.numerator);
      else
          BASIS_LOG_INFO("[{},{}] fps", 1.0*frmival.stepwise.max.denominator/frmival.stepwise.max.numerator, 1.0*frmival.stepwise.min.denominator/frmival.stepwise.min.numerator);
      frmival.index++;    
  }

  BASIS_LOG_INFO("Current frame rate {}/{}", streamparm.parm.capture.timeperframe.numerator, streamparm.parm.capture.timeperframe.denominator);
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = 30;
  BASIS_LOG_INFO("Setting frame rate to {}/{}...", streamparm.parm.capture.timeperframe.numerator, streamparm.parm.capture.timeperframe.denominator);
  if (ioctl(fd, VIDIOC_S_PARM, &streamparm) < 0) {
    BASIS_LOG_ERROR("Failed to VIDIOC_S_PARM");
    close(fd);
    return false;
  }

  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (ioctl(fd, VIDIOC_G_PARM, &streamparm) < 0) {
    BASIS_LOG_ERROR("VIDIOC_G_PARM");
    close(fd);
    return false;
  }

  BASIS_LOG_INFO("Current frame rate {}/{}", streamparm.parm.capture.timeperframe.numerator, streamparm.parm.capture.timeperframe.denominator);
  
  // 4. Request Buffers from the device
  v4l2_requestbuffers requestBuffer = {0};
  requestBuffer.count = BUFFER_COUNT;                          
  requestBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // request a buffer wich we an use for capturing frames
  requestBuffer.memory = V4L2_MEMORY_MMAP;

BASIS_LOG_INFO("Request {} buffers", BUFFER_COUNT);
  if (ioctl(fd, VIDIOC_REQBUFS, &requestBuffer) < 0) {
    BASIS_LOG_ERROR("Could not request buffer from device, VIDIOC_REQBUFS");
    close(fd);
    return false;
  }
  for(int buffer_index = 0; buffer_index < BUFFER_COUNT; buffer_index++) {
    // 5. Query the buffer to get raw data ie. ask for the you requested buffer
    // and allocate memory for it
    v4l2_buffer queryBuffer = {0};
    queryBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    queryBuffer.memory = V4L2_MEMORY_MMAP;
    queryBuffer.index = buffer_index;
            

    if (ioctl(fd, VIDIOC_QUERYBUF, &queryBuffer) < 0) {
      BASIS_LOG_ERROR("Device did not return the buffer information, VIDIOC_QUERYBUF");
      close(fd);
      return false;
    }
        

    // use a pointer to point to the newly created buffer
    // mmap() will map the memory address of the device to
    // an address in memory
    camera_buffers[buffer_index] =
        (char *)mmap(NULL, queryBuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, queryBuffer.m.offset);
    memset(camera_buffers[buffer_index], 0, queryBuffer.length);

    // 6. Get a frame
    // Create a new buffer type so the device knows whichbuffer we are talking about
    auto* buffer_info = buffer_infos + buffer_index;
    memset(buffer_info, 0, sizeof(v4l2_buffer));
    buffer_info->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer_info->memory = V4L2_MEMORY_MMAP;
    buffer_info->index = buffer_index;
  }
  // Activate streaming
  if (ioctl(fd, VIDIOC_STREAMON, &buffer_infos[0].type) < 0) {
    BASIS_LOG_ERROR("Could not start streaming, VIDIOC_STREAMON");
    close(fd);
    return false;
  }

  BASIS_LOG_INFO("Successfully opened camera at {}", camera_device);



  camera_fd = fd;

  for(int i = 0; i < BUFFER_COUNT; i++) {
    if (!Queue(i)) {
      BASIS_LOG_ERROR("Could not queue buffer, VIDIOC_QBUF");
      CloseCamera();
      return false;
    }
  }

  return true;
}

bool v4l2_camera_driver::Queue(int index) {
  return ioctl(camera_fd, VIDIOC_QBUF, buffer_infos + index) >= 0;
}
bool v4l2_camera_driver::Dequeue(int index) {
  return ioctl(camera_fd, VIDIOC_DQBUF, buffer_infos + index) >= 0;
}

// TODO: move this to another thread?
OnCameraImage::Output v4l2_camera_driver::OnCameraImage(const OnCameraImage::Input &input) {
  OnCameraImage::Output output;
  if (InitializeCamera("/dev/video0")) {
    current_index++;
    current_index %= BUFFER_COUNT;

    // Dequeue the buffer
    if (!Dequeue(current_index)) {
      BASIS_LOG_ERROR("Could not dequeue the buffer, VIDIOC_DQBUF");
      CloseCamera();
      return {};
    }

    output.camera_yuyv_cuda = std::make_shared<image_conversion::CudaManagedImage>(image_conversion::PixelFormat::YUV422, (size_t)imageFormat.fmt.pix.width, (size_t)imageFormat.fmt.pix.height,  input.time, (std::byte*)camera_buffers[current_index]);
    output.camera_yuyv = output.camera_yuyv_cuda->ToFoxglove();

    if (!Queue(current_index)) {
      BASIS_LOG_ERROR("Could not queue buffer, VIDIOC_QBUF");
      CloseCamera();
    }
  }
  return output;
}

void v4l2_camera_driver::CloseCamera() {
  if (camera_fd > -1) {

    // end streaming
    if (ioctl(camera_fd, VIDIOC_STREAMOFF, &buffer_infos[0].type) < 0) {
      BASIS_LOG_ERROR("Could not end streaming, VIDIOC_STREAMOFF");
    }
    close(camera_fd);
    camera_fd = -1;
  }
}