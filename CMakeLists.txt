#****************************************************************************#
#       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           #
#                                                                            #
#  Distributed under the terms of the GNU General Public License (GPL)       #
#                                                                            #
#    This code is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       #
#   General Public License for more details.                                 #
#                                                                            #
#  The full text of the GPL is available at:                                 #
#                                                                            #
#                  http://www.gnu.org/licenses/                               #
#****************************************************************************#

cmake_minimum_required(VERSION 3.5.1) # Support of nvcc

#####################
# Project description
project(HPCombi)

set(DESCRIPTION  "High Performance Combinatorics in C++ using vector instructions" CACHE STRING "Project description.")
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   2   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "**** Build type = ${CMAKE_BUILD_TYPE}")
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_VERBOSE_MAKEFILE 1)


################################
# General compiler configuration
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++11 instead of -std=gnu++11

add_definitions(-DHPCOMBI_HAVE_CONFIG)

option(USE_CUDA "Use CUDA" ON)
if(USE_CUDA)
	add_definitions(-DCOMPILE_CUDA)
endif ( )

message(STATUS "*** Compiler id is ${CMAKE_CXX_COMPILER_ID}")
if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  # Workaround of CMAKE bug https://stackoverflow.com/questions/47213356/
  set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -std=c++11)
  add_compile_options( -std=c++11 )
endif ( )

###################
# Project Structure
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmark)
add_subdirectory(doc)


include_directories(include ${PROJECT_BINARY_DIR})


#########
# Testing

IF (BUILD_TESTING)
include(CTest)
enable_testing ()
add_subdirectory(tests)
ENDIF(BUILD_TESTING)

#####################
# config.h file stuff
configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/HPCombi-config.h)
# install (FILES ${CMAKE_CURRENT_BINARY_DIR}/HPCombi-config.h
#   DESTINATION include/${CMAKE_PROJECT_NAME})
configure_file(${CMAKE_SOURCE_DIR}/VERSION.in ${CMAKE_BINARY_DIR}/VERSION)

####################
# Install Misc
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/VERSION DESTINATION ".")
install (FILES ${CMAKE_SOURCE_DIR}/README.md DESTINATION ".")
install (FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION ".")
install (FILES ${CMAKE_SOURCE_DIR}/list_intrin.txt DESTINATION ".")
###################
# pkgconfig stuff
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hpcombi.pc.in
#                ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc @ONLY)


#################
# Packing stuff
#
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${DESCRIPTION}")
set(CPACK_PACKAGE_VENDOR "Florent Hivert <florent.hivert@lri.fr>")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")

set(CPACK_GENERATOR "TGZ")
SET(CPACK_PACKAGE_FILE_NAME
  "HPCombi-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
include(CPack)

########################
# Custom target for TAGS
if (UNIX)
  add_custom_target(tags etags --members --declarations  `find ${CMAKE_SOURCE_DIR}/ -name *.cpp -or -name *.hpp -or -name *.c -or -name *.h` -o ${CMAKE_SOURCE_DIR}/TAGS)
  add_custom_target(etags DEPENDS tags)
endif (UNIX)



