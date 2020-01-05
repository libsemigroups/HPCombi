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

include (CheckIncludeFileCXX)
check_include_file_cxx("sparsehash/dense_hash_map" HPCOMBI_HAVE_DENSEHASHMAP)
if (NOT HPCOMBI_HAVE_DENSEHASHMAP)
  message(STATUS "Google 'dense_hash_map' not found, using std::unordered_map")
endif()
check_include_file_cxx("sparsehash/dense_hash_set" HPCOMBI_HAVE_DENSEHASHSET)
if (NOT HPCOMBI_HAVE_DENSEHASHSET)
  message(STATUS "Google 'dense_hash_set' not found, using std::unordered_set")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})

set(usage_examples RD.cpp Renner.cpp stringmonoid.cpp Trans.cpp CF.cpp image.cpp)

foreach(f ${usage_examples})
  get_filename_component(exampleName ${f} NAME_WE)
  add_executable (${exampleName} ${f})
  target_compile_options(${exampleName} PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${exampleName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${exampleName})
endforeach(f)
