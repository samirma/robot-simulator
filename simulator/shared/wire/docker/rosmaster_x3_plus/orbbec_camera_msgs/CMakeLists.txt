# Message-only build of the orbbec_camera package (its msg/ and srv/ come from the pinned
# OrbbecSDK_ROS1 release; the camera driver itself is simulated).
cmake_minimum_required(VERSION 3.0.2)
project(orbbec_camera)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs)
add_message_files(FILES DeviceInfo.msg Extrinsics.msg Metadata.msg)
add_service_files(FILES GetBool.srv SetBool.srv GetCameraInfo.srv GetCameraParams.srv
                  GetDeviceInfo.srv GetInt32.srv GetString.srv SetInt32.srv SetString.srv)
generate_messages(DEPENDENCIES std_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs)
