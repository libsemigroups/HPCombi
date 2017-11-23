IF (BUILD_TESTING)

find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

include_directories (
  ${CMAKE_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${Boost_INCLUDE_DIRS}
  )

add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable (test_perm16 test_perm16.cpp)
target_link_libraries(test_perm16
  perm16
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )
add_executable (test_perm_all test_perm_all.cpp)
target_link_libraries(test_perm_all
  perm16
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )

add_test (NAME TestPerm16 COMMAND test_perm16)
add_test (NAME TestPermAll COMMAND test_perm_all)

ENDIF(BUILD_TESTING)
