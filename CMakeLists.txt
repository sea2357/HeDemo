cmake_minimum_required(VERSION 3.10)
INCLUDE_DIRECTORIES(./include)
#INCLUDE_DIRECTORIES(/usr/include)
#LINK_DIRECTORIES(/usr/lib)

find_package(PkgConfig REQUIRED)
pkg_search_module(OPENCV REQUIRED opencv)

add_executable(main main.cpp knn.cpp)
TARGET_LINK_LIBRARIES(main ${OPENCV_LIBRARIES})
TARGET_LINK_LIBRARIES(main pthread)