include_directories (${CMAKE_SOURCE_DIR}/include)

add_library(perm16 SHARED perm16.cpp)
add_library(testtools SHARED testtools.cpp)

install (TARGETS testtools DESTINATION lib)
install (TARGETS perm16 DESTINATION lib)

file(
	GLOB 
	headers
	*.hpp
)

target_compile_options(perm16 PUBLIC -std=c++14 -O3 -Wall -mavx -mtune=native -funroll-loops -flax-vector-conversions)
target_compile_options(testtools PUBLIC -std=c++14 -O3 -Wall -mavx -mtune=native -funroll-loops -flax-vector-conversions)

install(FILES ${headers}
  DESTINATION include/${CMAKE_PROJECT_NAME})
