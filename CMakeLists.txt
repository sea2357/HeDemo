cmake_minimum_required(VERSION 3.12)
project(HeDemo  LANGUAGES CXX C)

list(APPEND CMAKE_PREFIX_PATH "./seal")
INCLUDE_DIRECTORIES(./include ./seal/include/SEAL-3.4/seal)

find_package(SEAL 3.4 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(OPENCV REQUIRED opencv)

add_executable(main  main.cpp knn.cpp)
target_link_libraries(main SEAL::seal)
TARGET_LINK_LIBRARIES(main
      ${OPENCV_LIBRARIES}
       pthread
)
