cmake_minimum_required(VERSION 3.13)
project(ORBextractor)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(ORBextractor main.cpp main.h ORBextractor.cpp ORBextractor.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})