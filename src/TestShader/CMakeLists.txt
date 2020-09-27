project (TestShader)

file(GLOB  SOURCES  *.h *.cpp)

add_executable(TestShader ${SOURCES})

###############################################
# Add .user file for Visual Studio to add bin to path
###############################################
if(${MSVC}) 
	configure_file(
		"${PROJECT_SOURCE_DIR}/vs_tmp.vcxproj.user.in" 
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}.vcxproj.user"
		@ONLY)
endif()
