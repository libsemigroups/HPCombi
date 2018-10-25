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


message(STATUS "Building tests")

find_package (Boost COMPONENTS unit_test_framework REQUIRED)

include_directories ( ${Boost_INCLUDE_DIRS} )
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/fallback
  ${PROJECT_BINARY_DIR})

add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable (test_epu test_epu.cpp)
target_link_libraries(test_epu ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
target_compile_options(test_epu PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)

add_executable (test_perm16 test_perm16.cpp)
target_link_libraries(test_perm16 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
target_compile_options(test_perm16 PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)

add_executable (test_perm_all test_perm_all.cpp)
target_link_libraries(test_perm_all ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
target_compile_options(test_perm_all PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)

add_executable (test_bmat8 test_bmat8.cpp)
target_link_libraries(test_bmat8 ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
target_compile_options(test_bmat8 PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)

add_executable (test_mincl test_mincl.cpp test_mincl0.cpp test_mincl1.cpp)
target_link_libraries(test_mincl ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
target_compile_options(test_mincl PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)

add_test (TestEPU test_epu)
add_test (TestPerm16 test_perm16)
add_test (TestPermAll test_perm_all)
add_test (TestBMat8 test_bmat8)
add_test (TestMultIncl test_mincl)

