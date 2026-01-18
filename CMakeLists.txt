cmake_minimum_required(VERSION 3.0.2)
project(Global-LVBA)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)           
find_package(GLUT REQUIRED)          
find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

set(Sophus_LIBRARIES libSophus.so)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES lvba_system lvba_io
  CATKIN_DEPENDS roscpp rospy std_msgs vikit_ros
  DEPENDS PCL OpenCV EIGEN3 
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SiftGPU/src/SiftGPU
)

set(SIFTGPU_LIBS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SiftGPU/build/src/SiftGPU/libsiftgpu.a
)
add_library(lvba_system src/lvba_system.cpp)
add_library(lvba_io src/dataset_io.cpp)

target_link_libraries(lvba_system
  PUBLIC
    Ceres::ceres
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${SIFTGPU_LIBS}
    GLEW::GLEW
    GLUT::GLUT 
    OpenGL::GL
)

target_link_libraries(lvba_io
  PUBLIC
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES}
)

add_executable(lidar_visual_ba src/main.cpp)
target_link_libraries(lidar_visual_ba
  PRIVATE
    lvba_system
    lvba_io
)
