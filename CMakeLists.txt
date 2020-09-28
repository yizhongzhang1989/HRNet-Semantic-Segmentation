cmake_minimum_required (VERSION 3.2)
cmake_policy(SET CMP0015 NEW)

project (CppTools LANGUAGES C CXX)

###############################################
# Options
###############################################
option(BUILD_Pano2Image "Whether build Pano2Image" ON)
option(BUILD_ReadPanorama "Whether build ReadPanorama" ON)
option(BUILD_TestShader "Whether build TestShader" ON)

set(GROUND_TEX_DIR ${PROJECT_SOURCE_DIR}/ground_tex)
configure_file(
  "${PROJECT_SOURCE_DIR}/data_config.h.in"
  "${PROJECT_BINARY_DIR}/data_config.h"
  @ONLY
  )

###############################################
# Debug & Output options
###############################################
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
	set(CMAKE_DEBUG_POSTFIX "d")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/lib)	 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin)


###############################################
# Dependency
###############################################

message(STATUS "==================================================")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BIN_PATH "${CMAKE_SOURCE_DIR}/3rdParty/bin/win64;${CMAKE_SOURCE_DIR}/3rdParty/bin/win32;${CMAKE_SOURCE_DIR}/bin/win64")

# --------------------
# Already included 3rdParty Library
# --------------------
include_directories(
	./3rdParty/inc
	${PROJECT_BINARY_DIR}
	)  

message(STATUS "==================================================")


###############################################
# path
###############################################
if(${MSVC}) 
	link_directories(./3rdParty/lib/win64 ${PROJECT_SOURCE_DIR}/lib/Release ${PROJECT_SOURCE_DIR}/lib/Debug)
elseif(${UNIX})
	link_directories(./3rdParty/lib/linux64 ${PROJECT_SOURCE_DIR}/bin/linux64)
	set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/linux64)
	set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/linux64)
endif(${MSVC})

include_directories(./src/common)


###############################################
# Source
###############################################
if (BUILD_Pano2Image)
	add_subdirectory(src/Pano2Image)
endif ()

if (BUILD_ReadPanorama)
	add_subdirectory(src/ReadPanorama)
endif ()

if (BUILD_TestShader)
	add_subdirectory(src/TestShader)
endif ()
