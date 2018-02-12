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

set(benchmark_src cycle.cpp inverse.cpp length.cpp sort.cpp sum.cpp bench.cpp)

foreach(f ${benchmark_src})
  get_filename_component(benchName ${f} NAME_WE)
  add_executable (${benchName} ${f})
  target_compile_options(${benchName} PUBLIC -Winline -mavx -mtune=native -funroll-loops -flax-vector-conversions)
  target_link_libraries(${benchName} testtools benchmark pthread)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${benchName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${benchName})
endforeach(f)

file(GLOB bench_gpu "${CMAKE_SOURCE_DIR}/benchmark/bench_gpu.cu")



if(USE_CUDA)
	get_filename_component(benchName ${bench_gpu} NAME_WE)
	if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
	        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
		# Modern CMake
        enable_language("CUDA")
		add_executable (${benchName} ${bench_gpu})
		add_compile_options( -gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_61,code=compute_61 )
		target_compile_options(${benchName} PUBLIC -Xcompiler -Winline,-mavx,-mtune=native,-funroll-loops,-flax-vector-conversions)
		target_link_libraries(${benchName} benchmark pthread gpu)

	else()
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
			add_compile_options( -gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_61,code=compute_61 )
			target_compile_options(${benchName} PUBLIC -Xcompiler -Winline,-mavx,-mtune=native,-funroll-loops,-flax-vector-conversions)
			target_link_libraries(${benchName} benchmark pthread gpu)
            CUDA_ADD_LIBRARY(${benchName} SHARED ${bench_gpu})
        endif()
    endif()
endif()
