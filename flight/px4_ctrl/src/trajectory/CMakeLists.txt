cmake_minimum_required(VERSION 3.0.2)
project(trajectory)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Wall -Wextra -fsanitize=address,undefined -static-libasan")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  quadrotor_msgs
  message_filters
)

find_package(Eigen3 REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

catkin_package(
  INCLUDE_DIRS include
  #  LIBRARIES trajectory
  #  CATKIN_DEPENDS roscpp
  #  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(trajectory 
  SHARED
  src/trajectory/trajectory.cpp
  src/trajectory/simulation.cpp
  src/trajectory/common.cpp
  src/trajectory/visual.cpp)
target_link_libraries(trajectory
  minco
  ${catkin_LIBRARIES}
  Eigen3::Eigen)

add_library(minco
  SHARED
  src/minco/traj_opt.cpp)
target_include_directories(minco PUBLIC
  ${PROJECT_SOURCE_DIR}/include/minco)
target_link_libraries(minco 
  ${catkin_LIBRARIES}
  Eigen3::Eigen)

add_executable(traj_gen
  src/traj_gen.cpp)
target_link_libraries(traj_gen
  minco
  trajectory
  ${catkin_LIBRARIES}
  Eigen3::Eigen
)

add_executable(traj_test
  src/traj_test.cpp)
target_link_libraries(traj_test
  minco
  trajectory
  ${catkin_LIBRARIES}
  Eigen3::Eigen
)