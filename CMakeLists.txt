cmake_minimum_required(VERSION 2.8)
project(BilliardEye)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}/opencv4)
include_directories(./include)

add_executable(segBill ./src/main.cpp)
target_link_libraries(segBill ${OpenCV_LIBS} )
