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

cmake_minimum_required(VERSION 2.8)

#####################
# Project description
project(HPCombi)

set(DESCRIPTION  "High Performance Combinatorics in C++ using vector instructions" CACHE STRING "Project description.")
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "**** Build type = ${CMAKE_BUILD_TYPE}")
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_VERBOSE_MAKEFILE 1)


################################
# General compiler configuration
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++11 instead of -std=gnu++11

message(STATUS "*** Compiler id is ${CMAKE_CXX_COMPILER_ID}")
if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  # Workaround of CMAKE bug https://stackoverflow.com/questions/47213356/
  set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -std=c++14)
  add_compile_options( -std=c++14 -Wall )
endif ( )
if ( MSVC )
   add_compile_options( /W4 )
endif ( MSVC )


###################
# Project Structure
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmark)
add_subdirectory(doc)

add_subdirectory(tests)

include_directories(include ${PROJECT_BINARY_DIR})


#####################
# config.h file stuff
configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/HPCombi-config.h)

###################
# pkgconfig stuff
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hpcombi.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc @ONLY)
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc
         DESTINATION lib/pkgconfig/)


########################
# Custom target for TAGS
if (UNIX)
  add_custom_target(tags etags --members --declarations  `find ${CMAKE_SOURCE_DIR}/ -name *.cpp -or -name *.hpp -or -name *.c -or -name *.h` -o ${CMAKE_SOURCE_DIR}/TAGS)
  add_custom_target(etags DEPENDS tags)
endif (UNIX)



