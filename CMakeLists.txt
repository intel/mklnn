CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

#project(mklEngine)

IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing Torch-mklnn through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()



FIND_PACKAGE(Torch REQUIRED)


FILE(GLOB luasrc ./*.lua)
FILE(GLOB src ./build/*.c)
#SET(src "init.c")
MESSAGE(STATUS "src =" "${src}")
ADD_TORCH_PACKAGE(mklnn "${src}" "${luasrc}" "Intel MKLNN")


