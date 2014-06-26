project(PANORAMA)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

find_package(OpenCV REQUIRED)

#vision executable
add_executable (main main.cpp detectBodyParts.cpp)
target_link_libraries(main ${OpenCV_LIBS})
