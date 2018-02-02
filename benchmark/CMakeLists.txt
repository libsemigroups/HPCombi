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

include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})

#~ file(GLOB benchmark_src "${CMAKE_SOURCE_DIR}/benchmark/*.cpp")
set(benchmark_src cycle.cpp inverse.cpp length.cpp sort.cpp sum.cpp bench.cpp)
set(benchmark_gpu_src bench_gpu.cpp)

foreach(f ${benchmark_src})
  get_filename_component(benchName ${f} NAME_WE)
  add_executable (${benchName} ${f})
  target_compile_options(${benchName} PUBLIC -Winline -mavx -mtune=native -funroll-loops -flax-vector-conversions)
  target_link_libraries(${benchName} testtools benchmark pthread)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${benchName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${benchName})
endforeach(f)


if(USE_CUDA)
	foreach(f ${benchmark_gpu_src})
		get_filename_component(benchName ${f} NAME_WE)
		add_executable (${benchName} ${f})
		target_compile_options(${benchName} PUBLIC -Winline -mavx -mtune=native -funroll-loops -flax-vector-conversions)
		target_link_libraries(${benchName} testtools benchmark pthread gpu)
	endforeach(f)
endif()
