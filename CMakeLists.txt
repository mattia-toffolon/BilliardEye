cmake_minimum_required(VERSION 2.8)
project(BilliardEye)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}/opencv4)
include_directories(./include)

add_library(segTable 
    ./include/segment/segTable.h
    ./src/segment/segTable.cpp
)
target_include_directories(segTable PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(ballType
    ./include/recognition/ballType.h
    ./src/recognition/ballType.cpp
)

add_executable(segBill ./src/main.cpp)
target_link_libraries(segBill segTable ${OpenCV_LIBS} )
target_link_libraries(segBill ballType ${OpenCV_LIBS})
