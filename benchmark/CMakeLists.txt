include_directories (${CMAKE_SOURCE_DIR}/include)

set(benchmark_src cycle.cpp inverse.cpp length.cpp sort.cpp sum.cpp)

foreach(f ${benchmark_src})
  get_filename_component(benchName ${f} NAME_WE)
  add_executable (${benchName} ${f})
  target_link_libraries(${benchName} perm16 testtools)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${benchName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${benchName})
endforeach(f)
