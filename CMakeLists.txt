cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

set (CMAKE_CXX_STANDARD 14)

project(HPCombi)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmark)
add_subdirectory(doc)

add_subdirectory(tests)
enable_testing ()
add_test (NAME TestPerm16 COMMAND test_perm16)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/hpcombi.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc @ONLY)
install (FILES ${CMAKE_CURRENT_BINARY_DIR}/hpcombi.pc
         DESTINATION lib/pkgconfig/)

IF (UNIX)
  ADD_CUSTOM_TARGET(tags etags --members --declarations  `find ${CMAKE_SOURCE_DIR}/ -name *.cpp -or -name *.hpp -or -name *.c -or -name *.h` -o ${CMAKE_SOURCE_DIR}/TAGS)
  ADD_CUSTOM_TARGET(etags DEPENDS tags)
ENDIF (UNIX)
