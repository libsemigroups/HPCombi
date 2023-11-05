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

set(test_src
  test_epu8.cpp test_xpu8.cpp test_perm16.cpp test_perm_all.cpp test_bmat8.cpp)

foreach(f ${test_src})
  get_filename_component(testName ${f} NAME_WE)
  add_executable (${testName} ${f} test_main.cpp)
  target_link_libraries(${testName} PRIVATE Catch2::Catch2WithMain)
endforeach(f)

add_executable(test_all ${test_src} test_main.cpp)
target_link_libraries(test_all PRIVATE Catch2::Catch2WithMain)

add_test (TestEPU8 test_epu8)
add_test (TestXPU8 test_xpu8)
add_test (TestPerm16 test_perm16)
add_test (TestPermAll test_perm_all)
add_test (TestBMat8 test_bmat8)
