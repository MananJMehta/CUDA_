cmake_minimum_required(VERSION 2.8)
project( cascadeclassifier )
set( OpenCV_FOUND 1 )
set( OpenCV_DIR "/home/ubuntu/opencv-3.1.0/build")
find_package( OpenCV REQUIRED )
add_executable( cascadeclassifier cascadeclassifier.cpp)
target_link_libraries( cascadeclassifier ${OpenCV_LIBS} )
