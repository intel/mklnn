CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

#project(mklEngine)

IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing Torch-mklnn through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()



FIND_PACKAGE(Torch REQUIRED)

#SET(SRC test.c SpatialConvolutionMKLDNN.c)
#file(GLOB SRC "./*.c")

#add_library(mklEngine SHARED ${SRC})

#install (TARGETS mklEngine DESTINATION ~/torch-intel/extra/mklnn/build/lib/mklEngine)
#install (FILES ../SpatialConvolution.lua DESTINATION ~/torch-intel/extra/mklnn)

FILE(GLOB luasrc ../*.lua)
SET(src "")
ADD_TORCH_PACKAGE(mklnn "${src}" "${luasrc}" "Intel MKLNN")


#install (TARGETS mklEngine DESTINATION INST_LIBDIR)
#install (FILES ../*.lua DESTINATION INST_LUADIR)
