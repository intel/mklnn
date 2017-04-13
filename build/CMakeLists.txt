CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

project(mklEngine)
FIND_PACKAGE(Torch REQUIRED)

#SET(SRC test.c SpatialConvolutionMKLDNN.c)
file(GLOB SRC "./*.c")

add_library(mklEngine SHARED ${SRC})

install (TARGETS mklEngine DESTINATION ~/torch-intel/extra/mklnn/build/lib/mklEngine)
install (FILES ../SpatialConvolution.lua DESTINATION ~/torch-intel/extra/mklnn)
