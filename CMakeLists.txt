cmake_minimum_required(VERSION 3.14)
project(digeiz-corrupted)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(digeiz-corrupted main.cpp)

target_link_libraries(digeiz-corrupted ${OpenCV_LIBS})
