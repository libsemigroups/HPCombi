add_library(perm16
  SHARED
  perm16.cpp)

install (TARGETS perm16
	 DESTINATION lib)

file(
	GLOB 
	headers
	*.hpp
)

target_compile_options(perm16 PUBLIC -std=c++14 -Wall -mavx -mtune=native -funroll-loops -flax-vector-conversions)

install(FILES ${headers}
  DESTINATION include/${CMAKE_PROJECT_NAME})
