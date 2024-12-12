/*

  This is the starting point for your Unit. Edit this directly and implement the missing methods!

*/
#include <unit/joystick_driver/unit_base.h>

#include <libevdev/libevdev.h>

struct Joystick
{
	Joystick(const char* path);
	~Joystick();

	Joystick(Joystick const&) = delete;
    Joystick& operator=(Joystick const&) = delete;

	Joystick(Joystick && other) {
		*this = std::move(other);
	}
	Joystick& operator=(Joystick && other) {
		std::swap(dev, other.dev);
		std::swap(fd, other.fd);
		buttons = other.buttons;
		axes_info = other.axes_info;
		path = other.path;
		name = other.name;
		return *this;
	}

	void Reset() {
		buttons = 0;
		axes_normalized = {};
	}

	static const unsigned int maxButtons = BTN_THUMBR - BTN_JOYSTICK + 1;
	static const unsigned int maxAxes = ABS_HAT3Y - ABS_X + 1;

    libevdev *dev = nullptr;
	int fd = -1;

	uint32_t buttons = 0;
	std::array<input_absinfo, maxAxes> axes_info = {};
	std::array<float, maxAxes> axes_normalized = {};
	std::string path;
	std::string name;
};

static_assert(Joystick::maxButtons <= 32);


class joystick_driver : public unit::joystick_driver::Base {
public:
  joystick_driver(const Args& args, const std::optional<std::string_view>& name_override = {});
  ~joystick_driver();

  virtual unit::joystick_driver::Tick::Output
  Tick(const unit::joystick_driver::Tick::Input &input) override;

  void readJoystickInput(Joystick* joystick);
  void openJoysticks();
  void closeJoysticks();

  std::vector<Joystick> joysticks;
  int inotify_fd = -1;
};