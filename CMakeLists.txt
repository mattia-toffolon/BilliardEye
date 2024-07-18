cmake_minimum_required(VERSION 2.8)
project(BilliardEye)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}/opencv4)
include_directories(./include)

add_library(segTable 
    ./include/segment/segTable.h
    ./src/segment/segTable.cpp
)
target_include_directories(segTable PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(segBalls 
    ./include/segment/segBalls.h
    ./src/segment/segBalls.cpp
)
target_include_directories(segBalls PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(ballIdentifier
    ./include/recognition/ballIdentifier.h
    ./src/recognition/ballIdentifier.cpp
)
target_include_directories(ballIdentifier PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(videoReader
    ./include/utils/VideoReader.hpp
    ./src/utils/VideoReader.cpp
)
target_include_directories(videoReader PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(balls
    ./include/utils/balls.hpp
    ./src/utils/balls.cpp
)
target_include_directories(balls PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(bboxesUtils
    ./include/utils/bboxesUtils.hpp
    ./src/utils/bboxesUtils.cpp
)
target_include_directories(bboxesUtils PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(side_recognition
    ./include/recognition/side_recognition.hpp
    ./src/recognition/side_recognition.cpp
)
target_include_directories(side_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(getTableColor
    ./include/utils/getTableColor.hpp
    ./src/utils/getTableColor.cpp
)
target_include_directories(getTableColor PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(ball_tracking
    ./include/tracking/TrackBalls.hpp
    ./src/tracking/TrackBalls.cpp
)
target_include_directories(ball_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(rendering
    ./include/rendering/render_table.hpp
    ./src/rendering/render_table.cpp
)
target_include_directories(rendering PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(table_transform
    ./include/recognition/transformPoints.hpp
    ./src/recognition/transformPoints.cpp
)
target_include_directories(table_transform PRIVATE ${OpenCV_INCLUDE_DIRS})

add_library(perfTesting
    ./include/utils/perfTesting.h
    ./src/utils/perfTesting.cpp
)
target_include_directories(perfTesting PRIVATE ${OpenCV_INCLUDE_DIRS})

# add_executable(segBill ./src/main.cpp)
add_executable(segBill ./src/shared_main.cpp)
add_executable(perfBill ./src/performance_main.cpp)
target_link_libraries(segBill
    segTable 
    segBalls
    ballIdentifier
    videoReader
    bboxesUtils
    getTableColor
    side_recognition
    balls
    table_transform
    ball_tracking 
    rendering 
    perfTesting
    ${OpenCV_LIBS}
    )
target_link_libraries(perfBill
    perfTesting
    balls
    bboxesUtils
    ${OpenCV_LIBS})
