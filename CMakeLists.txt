cmake_minimum_required(VERSION 3.13)
project(ORBextractor)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

find_package(OpenCV REQUIRED)

include_directories(.)

add_executable(ORBextractor src/main.cpp include/main.h src/ORBextractor.cpp include/ORBextractor.h
        src/referenceORB.cpp include/referenceORB.h src/Distribution.cpp include/Distribution.h
        src/CVFAST.cpp include/CVFAST.h include/ORBconstants.h include/KdTree.h include/Nanoflann.h include/RangeTree.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})