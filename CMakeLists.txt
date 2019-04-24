cmake_minimum_required(VERSION 3.13)
project(ORBextractor)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

include_directories(.)

add_executable(ORBextractor src/main.cpp include/main.h src/ORBextractor.cpp include/ORBextractor.h src/Comparison_Descriptors.cpp include/Comparison_Descriptors.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})