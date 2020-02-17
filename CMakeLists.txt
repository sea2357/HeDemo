cmake_minimum_required(VERSION 3.10)
INCLUDE_DIRECTORIES(./include)
#INCLUDE_DIRECTORIES(/usr/include)
#LINK_DIRECTORIES(/usr/lib)

add_executable(main main.cpp)
TARGET_LINK_LIBRARIES(main libopencv_core.a libtbb.so)
TARGET_LINK_LIBRARIES(main pthread libdl.so libz.so)