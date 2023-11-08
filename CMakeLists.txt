#****************************************************************************#
#  Copyright (C) 2016-2023 Florent Hivert <Florent.Hivert@lri.fr>,           #
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
#                  http://www.gnu.org/licenses/                              #
#****************************************************************************#

# Require at least 3.8 so that we can set cmake_policy CMP0067 below
cmake_minimum_required(VERSION 3.8)

#####################
# Project description
project(HPCombi)

set(DESCRIPTION  "High Performance Combinatorics in C++ using vector instructions"
  CACHE STRING "Project description.")
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   6   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "**** Build type = ${CMAKE_BUILD_TYPE}")
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_VERBOSE_MAKEFILE 1)

################################
# General compiler configuration
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++14 instead of -std=gnu++14

message(STATUS "*** Compiler id is ${CMAKE_CXX_COMPILER_ID}")
# See https://stackoverflow.com/questions/47213356/ for discussion
cmake_policy(SET CMP0067 NEW)

##################################
# Checks for compiler capabilities
include(CheckCXXSymbolExists)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

check_cxx_compiler_flag('-funroll-loops' HPCOMBI_HAVE_REQUIRED_COMPILER_FLAGS)
check_cxx_compiler_flag('-flax-vector-conversions' HPCOMBI_HAVE_REQUIRED_COMPILER_FLAGS)

if (NOT HPCOMBI_HAVE_REQUIRED_COMPILER_FLAGS)
  message(FATAL_ERROR "HPCombi requires a C++ compiler accepting the flags: -funroll-loops + -flax-vector-conversions")
endif()

add_compile_options(-funroll-loops -flax-vector-conversions)

file(READ ${CMAKE_SOURCE_DIR}/list_builtin.txt hpcombi_compiler_builtins)
string(REPLACE ";" "|" hpcombi_compiler_builtins "${hpcombi_compiler_builtins}")
string(REPLACE "\n" ";" hpcombi_compiler_builtins "${hpcombi_compiler_builtins}")
foreach (builtin ${hpcombi_compiler_builtins})
  if ("${builtin}" MATCHES "^#" ) # Comment
    continue()
  endif()
  string(REPLACE "|" ";" builtin "${builtin}")
  list(GET builtin 0 builtin_name)
  list(GET builtin 1 builtin_params)
  list(GET builtin 2 builtin_return_value)
  check_cxx_source_runs("
  #include <iostream>
    int main() {
      std::cout << \"<parameter;return value> from list_builtin.txt: \" << ${builtin_params} << \";\" << ${builtin_return_value} << std::endl;
      std::cout << \"Actual return value: \" << ${builtin_name}(${builtin_params}) << std::endl;
      return ${builtin_name}(${builtin_params}) != ${builtin_return_value};
    }
  "
  "HPCOMBI_HAVE_${builtin_name}"
  )
  if (NOT "${HPCOMBI_HAVE_${builtin_name}}")
    message(FATAL_ERROR "HPCombi requires a C++ compiler supporting ${builtin_name}")
  endif()
endforeach()

###################
# Project Structure
add_subdirectory(examples)
add_subdirectory(doc)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party
  ${PROJECT_BINARY_DIR})

#########
# Testing

if (BUILD_TESTING)
  include(CTest)
  enable_testing ()
  add_subdirectory(tests)
  add_subdirectory(benchmark)
endif(BUILD_TESTING)

#####################

configure_file(${CMAKE_SOURCE_DIR}/.VERSION.in ${CMAKE_BINARY_DIR}/.VERSION)

####################
# Installation

set(install_misc README.md LICENSE list_intrin.txt)
foreach(f ${install_misc})
  install (FILES ${CMAKE_SOURCE_DIR}/${f}
    DESTINATION share/${CMAKE_PROJECT_NAME})
endforeach(f)

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/.VERSION DESTINATION ".")

install (
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION include/${CMAKE_PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp")
install (
  DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/
  DESTINATION third_party/${CMAKE_PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp")

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
