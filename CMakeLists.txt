cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++11 instead of -std=gnu++11

if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif ( CMAKE_COMPILER_IS_GNUCC )
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif ( MSVC )

project(HPCombi)

set(DESCRIPTION  "High Performance Combinatorics in C++ using vector instructions" CACHE STRING "Project description.")
set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   0   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   1   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmark)
add_subdirectory(doc)

include_directories(include)

add_subdirectory(tests)
enable_testing ()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/hpcombi.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc @ONLY)
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc
         DESTINATION lib/pkgconfig/)

IF (UNIX)
  ADD_CUSTOM_TARGET(tags etags --members --declarations  `find ${CMAKE_SOURCE_DIR}/ -name *.cpp -or -name *.hpp -or -name *.c -or -name *.h` -o ${CMAKE_SOURCE_DIR}/TAGS)
  ADD_CUSTOM_TARGET(etags DEPENDS tags)
ENDIF (UNIX)
