include_directories (${CMAKE_SOURCE_DIR}/src)

INCLUDE (CheckIncludeFileCXX)
INCLUDE (CheckCXXSymbolExists)
CHECK_INCLUDE_FILE_CXX("sparsehash/dense_hash_map" HAVE_SPARSEHASH)
CHECK_CXX_SYMBOL_EXISTS("google::dense_hash_map" "sparsehash/dense_hash_map" HAVE_GOOGLE_HASH_DENSE_HASH_MAP)

file(
	GLOB
	usage_examples
	*.cpp
)

# set_target_properties(sometarget PROPERTIES CXX_STANDARD 11 CXX_EXTENSIONS FALSE CXX_STANDARD_REQUIRED TRUE)

foreach(f ${usage_examples})
  get_filename_component(exampleName ${f} NAME_WE) 
  add_executable (${exampleName} ${f}) 
  target_link_libraries(${exampleName} perm16)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${exampleName}
    DESTINATION bin
    RENAME ${CMAKE_PROJECT_NAME}-${exampleName})
endforeach(f)
