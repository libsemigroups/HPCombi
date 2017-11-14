include_directories (${CMAKE_SOURCE_DIR}/include)

INCLUDE (CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX("sparsehash/dense_hash_map" HAVE_SPARSEHASHMAP)
CHECK_INCLUDE_FILE_CXX("sparsehash/dense_hash_set" HAVE_SPARSEHASHSET)

set(usage_examples RD.cpp Renner.cpp stringmonoid.cpp Trans.cpp CF.cpp)

# set_target_properties(sometarget PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS FALSE CXX_STANDARD_REQUIRED TRUE)

foreach(f ${usage_examples})
  get_filename_component(exampleName ${f} NAME_WE) 
  add_executable (${exampleName} ${f}) 
  target_link_libraries(${exampleName} perm16)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${exampleName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${exampleName})
endforeach(f)
