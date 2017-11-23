include_directories (${CMAKE_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

INCLUDE(CheckIncludeFileCXX)
INCLUDE(CheckCXXSourceCompiles)
CHECK_INCLUDE_FILE_CXX("experimental/numeric" HAVE_EXPERIMENTAL_NUMERIC)
CHECK_CXX_SOURCE_COMPILES(
        "
        #include <experimental/numeric>
        static_assert( std::experimental::lcm(4, 6) == 12, \"Buggy lcm\");
        int main() { }
        "
        HAVE_EXPERIMENTAL_NUMERIC_LCM)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/config.hpp.in ${CMAKE_BINARY_DIR}/include/config.hpp)

add_library(perm16 SHARED perm16.cpp)
add_library(testtools SHARED testtools.cpp)

install (TARGETS testtools DESTINATION lib)
install (TARGETS perm16 DESTINATION lib)

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  target_compile_options(perm16 PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)
  target_compile_options(testtools PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)
endif ()

install(FILES ${headers}
  DESTINATION include/${CMAKE_PROJECT_NAME})
