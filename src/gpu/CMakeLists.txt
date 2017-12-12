
include_directories(${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/gpu/*.cu")

if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
    if(USE_CUDA)
        enable_language("CUDA")
        add_compile_options( -arch=compute_30 -code=sm_30 )
		add_library(gpu SHARED ${gpu_source_files})
		
    endif()
else()
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            add_compile_options( -arch=compute_30 -code=sm_30 )
            CUDA_ADD_LIBRARY(gpu SHARED ${gpu_source_files})
        endif()
    endif()
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
