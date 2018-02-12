
include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/gpu/*.cu")
#~ file(GLOB bench_gpu "${CMAKE_SOURCE_DIR}/benchmark/bench_gpu.cpp")


if(USE_CUDA)
	if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
	        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
		# Modern CMake
        enable_language("CUDA")
        add_compile_options( -gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_61,code=compute_61 -Xcompiler -fPIC)#-Wno-deprecated-declarations) # Warning about __shfl instrucion deprecited in cuda 9
		add_library(gpu SHARED ${gpu_source_files})

	else()
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            add_compile_options( -gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_61,code=compute_61 -Xcompiler -fPIC)
            CUDA_ADD_LIBRARY(gpu SHARED ${gpu_source_files})
        endif()
    endif()
    add_definitions(-DUSE_CUDA)
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
