# HeDemo
This is a demo written in C++. We try to show you how to apply homomorphic encryption to machine learning.

How to install and run test.
1. cd HeDemo
2. mkdir build && cd build
3. cmake ..
4. make
5. ./main

How to generate documents.
1. cd HeDemo 
2. doxygen Doxyfile
Then the documents is generated in HeDemo/html.

Problems may be encountered.

First of all, please check CMake (>= 3.12), GNU G++ (>= 6.0) or Clang++ (>= 5.0).

1. CMake Error: CMake was unable to find a build program corresponding to "Unix Makefiles".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.

  $ sudo apt install build-essential


2. Could NOT find ZLIB

  $ sudo apt install zlib1g zlib1g-dev


3. Could NOT find PkgConfig

  $ sudo apt install pkg-config


4. None of the required 'opencv' found

  $ sudo apt install libopencv-dev


5. The CMAKE_C_COMPILER:  /usr/bin/cc  is not a full path to an existing compiler tool

  $sudo apt install g++


6. CMake Error at /opt/cmake-3.16.4-Linux-x86_64/share/cmake-3.16/Modules/CMakeDetermineSystem.cmake:185 (configure_file):  configure_file Problem configuring file

  $ rm -rf build
