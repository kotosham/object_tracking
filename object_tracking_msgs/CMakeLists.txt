cmake_minimum_required(VERSION 3.8)
project(object_tracking_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Candidate.msg"
  "msg/PlanStep.msg"
  "msg/Notes.msg"
  "action/SeekObject.action"
  "action/DetectTarget.action"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
