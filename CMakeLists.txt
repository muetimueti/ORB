cmake_minimum_required(VERSION 3.13)
project(ORBextractor)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fopenmp")

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(.)

add_executable(ORBextractor src/main.cpp include/main.h src/ORBextractor.cpp include/ORBextractor.h
        src/Distribution.cpp include/Distribution.h
        include/ORBconstants.h include/Nanoflann.h include/RangeTree.h src/FAST.cpp include/FAST.h include/avx.h
        include/FASTworker.h include/Types.h include/FeatureFileInterface.h src/FeatureFileInterface.cpp
        saigacpy/assert.h saigacpy/config.h saigacpy/imageBase.h saigacpy/assert.cpp saigacpy/opencv.h
        src/2Dimgeffects.cpp include/2Dimgeffects.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})