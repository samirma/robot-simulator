// The simulator's stand-in for feetech_ros2_driver/FeetechHardwareInterface.
//
// The SO-101 boot's robot_description names this ros2_control plugin; the simulator
// provides a plugin of the same name so the boot's robot_description, controller_manager
// and controllers run unchanged. Instead of the STS3215 bus on a serial port it exchanges
// the bus traffic with the wire's servo-bus process (robots/so101_bus.py) over UDP on
// localhost: write() sends every commanded goal (radians, as the real plugin converts to
// ticks), read() takes the latest present position and speed of every servo, quantised
// to the STS3215's 4096 ticks per revolution exactly as the real plugin reads them.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#if __has_include(<hardware_interface/hardware_interface/version.h>)
#include <hardware_interface/hardware_interface/version.h>
#else
#include <hardware_interface/version.h>
#endif

namespace feetech_ros2_driver {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace {
constexpr int kBusPort = 47001;    // servo-bus process listens here for goals
constexpr int kStatePort = 47002;  // this plugin listens here for present state
constexpr double kTicksPerRad = 4096.0 / (2.0 * M_PI);
double quantise(double rad) { return std::round(rad * kTicksPerRad) / kTicksPerRad; }
}  // namespace

class FeetechHardwareInterface : public hardware_interface::SystemInterface {
 public:
#if HARDWARE_INTERFACE_VERSION_GTE(4, 34, 0)
  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams& params) override {
    if (hardware_interface::SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    return setup();
  }
#else
  CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override {
    if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
      return CallbackReturn::ERROR;
    }
    return setup();
  }
#endif

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override {
    std::vector<hardware_interface::StateInterface> out;
    for (size_t i = 0; i < info_.joints.size(); i++) {
      out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &pos_[i]);
      out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &vel_[i]);
    }
    return out;
  }

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override {
    std::vector<hardware_interface::CommandInterface> out;
    for (size_t i = 0; i < info_.joints.size(); i++) {
      if (!info_.joints[i].command_interfaces.empty()) {
        out.emplace_back(info_.joints[i].name, hardware_interface::HW_IF_POSITION, &cmd_[i]);
      }
    }
    return out;
  }

  hardware_interface::return_type read(const rclcpp::Time&, const rclcpp::Duration&) override {
    char buf[4096];
    bool got = false;
    std::string last;
    while (true) {
      ssize_t n = recv(state_sock_, buf, sizeof(buf) - 1, 0);
      if (n <= 0) break;
      buf[n] = 0;
      last = buf;
      got = true;
    }
    if (got) {
      std::istringstream in(last);
      std::string tag;
      in >> tag;
      for (size_t i = 0; i < info_.joints.size() && in; i++) {
        double p, v;
        in >> p >> v;
        pos_[i] = quantise(p);
        vel_[i] = quantise(v);
      }
      have_state_ = true;
    }
    return hardware_interface::return_type::OK;
  }

  hardware_interface::return_type write(const rclcpp::Time&, const rclcpp::Duration&) override {
    std::ostringstream out;
    out.precision(9);
    out << "goal";
    for (size_t i = 0; i < info_.joints.size(); i++) {
      // The real plugin writes every commanded joint with speed 2400 and acceleration 50.
      out << " " << info_.joints[i].name << " " << (std::isnan(cmd_[i]) ? pos_[i] : quantise(cmd_[i]));
    }
    const std::string s = out.str();
    sendto(bus_sock_, s.data(), s.size(), 0, reinterpret_cast<sockaddr*>(&bus_addr_), sizeof(bus_addr_));
    return hardware_interface::return_type::OK;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
    // Wait for the bus to report every servo, then hold where they are.
    for (int i = 0; i < 200 && !have_state_; i++) {
      read(rclcpp::Time{}, rclcpp::Duration::from_seconds(0));
      if (!have_state_) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!have_state_) {
      RCLCPP_ERROR(rclcpp::get_logger("FeetechHardwareInterface"), "no servo reported on the bus");
      return CallbackReturn::ERROR;
    }
    cmd_ = pos_;
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override { return CallbackReturn::SUCCESS; }

 private:
  CallbackReturn setup() {
    if (info_.hardware_parameters.find("usb_port") == info_.hardware_parameters.end()) {
      RCLCPP_ERROR(rclcpp::get_logger("FeetechHardwareInterface"), "Hardware parameter [usb_port] not found");
      return CallbackReturn::ERROR;
    }
    pos_.assign(info_.joints.size(), 0.0);
    vel_.assign(info_.joints.size(), 0.0);
    cmd_.assign(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    bus_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    std::memset(&bus_addr_, 0, sizeof(bus_addr_));
    bus_addr_.sin_family = AF_INET;
    bus_addr_.sin_port = htons(kBusPort);
    bus_addr_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    state_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_port = htons(kStatePort);
    me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(state_sock_, reinterpret_cast<sockaddr*>(&me), sizeof(me)) != 0) {
      RCLCPP_ERROR(rclcpp::get_logger("FeetechHardwareInterface"), "cannot bind the bus state port");
      return CallbackReturn::ERROR;
    }
    fcntl(state_sock_, F_SETFL, O_NONBLOCK);
    return CallbackReturn::SUCCESS;
  }

  std::vector<double> pos_, vel_, cmd_;
  int bus_sock_{-1}, state_sock_{-1};
  sockaddr_in bus_addr_{};
  bool have_state_{false};
};

}  // namespace feetech_ros2_driver

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(feetech_ros2_driver::FeetechHardwareInterface, hardware_interface::SystemInterface)
