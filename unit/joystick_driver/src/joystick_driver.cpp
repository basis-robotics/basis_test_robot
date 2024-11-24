/*
  Based on https://github.com/MysteriousJ/Joystick-Input-Examples/blob/main/src/evdev.cpp
*/

#include <joystick_driver.h>

using namespace unit::joystick_driver;

#include <google/protobuf/util/time_util.h>

#include <libevdev/libevdev.h>


Joystick::Joystick(const char* path) : path(path) {
	fd = open(path, O_RDWR | O_NONBLOCK);
  if (fd == -1)
  {
    return;
  }

  int rc = libevdev_new_from_fd(fd, &dev);
    
  // Get name
  name = libevdev_get_name(dev);
}

Joystick::~Joystick() {
  libevdev_free(dev);
	close(fd);
}

joystick_driver::joystick_driver(const Args& args, const std::optional<std::string_view>& name_override) 
  : unit::joystick_driver::Base(args, name_override)
{
	openJoysticks();
	inotify_fd = inotify_init1(IN_NONBLOCK);
	inotify_add_watch(inotify_fd, "/dev/input", IN_ATTRIB);
}

joystick_driver::~joystick_driver() {
  closeJoysticks();
  close(inotify_fd);
}

void joystick_driver::openJoysticks()
{
	for (int i=0; i<32; ++i) {
    std::string fileName = "/dev/input/event" + std::to_string(i);
    BASIS_LOG_INFO("Opening joystick {}", fileName);

    Joystick j(fileName.c_str());
    if(j.fd == -1) {
      // Devices are always to be 0-N
      return;
    }

    if(!j.dev) {
      BASIS_LOG_INFO("Skipping joystick {}", fileName);
      continue;
    }

    BASIS_LOG_INFO("joystick {} is named {}", fileName, j.name);


    bool has_button_code = false;
    if(libevdev_has_event_type(j.dev, EV_KEY)) {
      for(int btn = BTN_JOYSTICK; btn <= BTN_THUMBR; btn++) {
        if(libevdev_has_event_code(j.dev, EV_KEY, btn)) {
          BASIS_LOG_INFO("Has event code");
          has_button_code = true;
          break;
        }
      }
    }    
    bool has_any_axis = false;
    // Setup axes
    for (unsigned int i=ABS_X; i<=ABS_HAT3Y; ++i)
    {
      const input_absinfo* axis_info = libevdev_get_abs_info(j.dev, i);
      if(axis_info) {
        has_any_axis = true;
        j.axes_info[i] = *axis_info;
      }
    }
    if(has_button_code || has_any_axis) {
      BASIS_LOG_INFO("Joystick {} added", fileName);
      joysticks.emplace_back(std::move(j));
    }
  }
}

void joystick_driver::closeJoysticks()
{
	joysticks.clear();
}

void joystick_driver::readJoystickInput(Joystick* joystick)
{
	input_event event;

  int flags = LIBEVDEV_READ_FLAG_NORMAL;

	while (true)
	{
    int rc = libevdev_next_event(joystick->dev, flags, &event);
    if(rc == -EAGAIN)
    {
      if(flags & LIBEVDEV_READ_FLAG_SYNC) {
            BASIS_LOG_ERROR("Joystick {} finished sync", joystick->name);
        flags &= ~LIBEVDEV_READ_FLAG_SYNC;
      }
      else {
        return;
      }
    }

    switch(event.type) {
      case EV_SYN:
        switch(event.code) {
          case SYN_DROPPED:
            BASIS_LOG_ERROR("Joystick {} dropped events, syncing...", joystick->name);
            // Reset all axes and buttons to zero
            joystick->Reset();
            break;
          default:
            // usually SYN_REPORT - batches events together
            break;
        }
        break;
      case EV_KEY: {
        const uint32_t button_mask = 1 << (event.code - BTN_JOYSTICK);
    
        if(event.value)  {
          joystick->buttons |= button_mask;
        }
        else {
          joystick->buttons &= ~button_mask;
        }
        break;
      }
      case EV_ABS: {
        if(event.code < ABS_TOOL_WIDTH) {
        	auto* axis = &joystick->axes_info[event.code];
			    float normalized = (event.value - axis->minimum) / (float)(axis->maximum - axis->minimum) * 2 - 1;
			    joystick->axes_normalized[event.code] = normalized;
        }
        break;
      }
    }
	}
}

// Experimental - create an arena allocated message
template<typename T>
struct ProtobufArenaMessage : public std::enable_shared_from_this<ProtobufArenaMessage<T>> {
  static std::shared_ptr<ProtobufArenaMessage<T>> New() {
    ProtobufArenaMessage out;

    auto self = std::shared_ptr<ProtobufArenaMessage<T>>(new ProtobufArenaMessage<T>());

    self->message = self->template CreateMessageShared<T>();

    return self;
  }

  template<typename T_MESSAGE>
  std::shared_ptr<T_MESSAGE> CreateMessageShared() {
    // The magic - makes a message owned by our arena, with lifetime tied to it
    return std::shared_ptr<T>(this->shared_from_this(), CreateMessage<T_MESSAGE>());
  }

  template<typename T_MESSAGE>
  T_MESSAGE* CreateMessage() {
    return google::protobuf::Arena::CreateMessage<T_MESSAGE>(&arena);
  }

  template<typename T_OTHER_TYPE, typename... T_ARGS>
  T_OTHER_TYPE* Create( T_ARGS&&... args) {
    return google::protobuf::Arena::Create<T>(&arena, std::forward(args)...);
  }

  std::shared_ptr<T>& operator->() {
    return message;
  }
  std::shared_ptr<T>& operator*() {
    return message;
  }

  operator std::shared_ptr<T>() const {
    return message;
  }

  operator std::shared_ptr<const T>() const {
    return std::const_pointer_cast<const T>(message);
  }
private:
  ProtobufArenaMessage() = default;
public:
  google::protobuf::Arena arena;
  std::shared_ptr<T> message;
};

Tick::Output joystick_driver::Tick(const Tick::Input& input) {
  // Update which joysticks are connected
  // Note: this may not actually work - should we remove it?
  inotify_event event;
  if (read(inotify_fd, &event, sizeof(event)+16) != -1)
  {
    closeJoysticks();
    openJoysticks();
  }

  // We don't need any additional arena access here, so we immediately discard the wrapper
  std::shared_ptr<basis::robot::input::InputState> inputs = **ProtobufArenaMessage<basis::robot::input::InputState>::New();
  const auto now = basis::core::MonotonicTime::Now();
  *inputs->mutable_timestamp() = google::protobuf::util::TimeUtil::NanosecondsToTimestamp(now.nsecs);

  // Update each connected joystick
  for (Joystick& joystick : joysticks)
  {
    readJoystickInput(&joystick);

    auto joystick_msg = inputs->mutable_joysticks()->Add();
    joystick_msg->set_name(joystick.name);
    joystick_msg->set_path(joystick.path);
    joystick_msg->set_buttons(joystick.buttons);
    joystick_msg->mutable_axes()->Add(joystick.axes_normalized.begin(), joystick.axes_normalized.end()); 
  }
  return {
    std::move(inputs)
  };
}
