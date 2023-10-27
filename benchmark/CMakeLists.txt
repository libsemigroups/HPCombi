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
#                  http://www.gnu.org/licenses/                              #
#****************************************************************************#

find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
  message(STATUS "Cloning Catch2 from GIT_REPOSITORY https://github.com/catchorg/Catch2.git")
  Include(FetchContent)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0 # or a later release
  )

  FetchContent_MakeAvailable(Catch2)
else()
  message(STATUS "Using system Catch2")
endif()

message(STATUS "Building tests")

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR})

set(benchmark_src
  bench_epu8.cpp bench_perm16.cpp bench_bmat8.cpp sort.cpp inverse.cpp)

foreach(f ${benchmark_src})
  get_filename_component(benchName ${f} NAME_WE)
  add_executable (${benchName} ${f})
  target_link_libraries(${benchName} PRIVATE Catch2::Catch2WithMain)
endforeach(f)
