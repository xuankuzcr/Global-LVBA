cmake_minimum_required(VERSION 3.5)
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
set(Ceres_DIR /media/shuai/deepslamer/MySfM/ceres-solver/install/lib/cmake/Ceres)
find_package(Ceres REQUIRED)

# 3D可视化依赖
find_package(glfw3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

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

# ============== GLAD ==============
# 需要手动下载glad源码到 thirdparty/glad 目录
# 从 https://glad.dav1d.de/ 生成 OpenGL 3.3 Core Profile
set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glad)
if(EXISTS ${GLAD_DIR}/src/glad.c)
    add_library(glad ${GLAD_DIR}/src/glad.c)
    target_include_directories(glad PUBLIC ${GLAD_DIR}/include)
else()
    message(WARNING "GLAD not found at ${GLAD_DIR}. Please download GLAD and place it in thirdparty/glad/")
endif()

# ============== Dear ImGui ==============
# 需要手动下载imgui源码到 thirdparty/imgui 目录
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui)
if(EXISTS ${IMGUI_DIR}/imgui.cpp)
    add_library(imgui
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC 
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
else()
    message(WARNING "Dear ImGui not found at ${IMGUI_DIR}. Please download ImGui and place it in thirdparty/imgui/")
endif()

# ============== 主项目库 ==============
add_library(lvba_system src/lvba_system.cpp)
add_library(lvba_io src/dataset_io.cpp)
add_library(pose_visualizer src/pose_visualizer.cpp)

# 3D可视化模块
target_link_libraries(pose_visualizer
  PUBLIC
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    glfw
    glad
    imgui
    OpenGL::GL
)

target_include_directories(pose_visualizer PUBLIC
    ${GLAD_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(lvba_system
  PUBLIC
    Ceres::ceres
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Sophus_LIBRARIES}
    ${SIFTGPU_LIBS}
    ${SQLITE3_LIBRARIES}
    GLEW::GLEW
    GLUT::GLUT 
    OpenGL::GL
    pose_visualizer
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
    pose_visualizer
    /usr/lib/x86_64-linux-gnu/libffi.so.7
)
