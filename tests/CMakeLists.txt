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

IF (BUILD_TESTING)

message(STATUS "Building tests")

find_package (Boost COMPONENTS unit_test_framework REQUIRED)

include_directories ( ${Boost_INCLUDE_DIRS} )
include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})

add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable (test_perm16 test_perm16.cpp)
target_link_libraries(test_perm16
  perm16
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )
add_executable (test_perm_all test_perm_all.cpp)
target_link_libraries(test_perm_all
  perm16
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )

include(CTest)
enable_testing ()

add_test (NAME TestPerm16 COMMAND test_perm16)
add_test (NAME TestPermAll COMMAND test_perm_all)

ENDIF(BUILD_TESTING)
