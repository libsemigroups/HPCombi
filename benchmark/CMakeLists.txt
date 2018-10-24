#****************************************************************************#
#       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           #
#                                                                            #
#  Distributed under the terms of the GNU General Public License (GPL)       #
#                                                                            #
#    This code is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       #
#   General Public License for more details.                                 #
#                                                                            #
#  The full text of the GPL is available at:                                 #
#                                                                            #
#                  http://www.gnu.org/licenses/                              #
#****************************************************************************#

include(CheckCXXSymbolExists)
include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)

check_include_file_cxx("experimental/numeric" HPCOMBI_HAVE_EXPERIMENTAL_NUMERIC)
if (HPCOMBI_HAVE_EXPERIMENTAL_NUMERIC)
  check_cxx_source_compiles(
        "
        #include <experimental/numeric>
        static_assert( std::experimental::lcm(4, 6) == 12, \"Buggy lcm\");
        int main() { }
        "
        HPCOMBI_HAVE_EXPERIMENTAL_NUMERIC_LCM)
endif (HPCOMBI_HAVE_EXPERIMENTAL_NUMERIC)

check_cxx_source_compiles(
  "
  using T = int;
  constexpr int exec(T f()) { return f(); }
  constexpr int foo() { return 1; }
  static_assert(exec(foo) == 1, \"Failed exec\");
  int main() {}
  "
  HPCOMBI_CONSTEXPR_FUN_ARGS)

check_include_file_cxx("x86intrin.h" HPCOMBI_HAVE_X86INTRIN)
if (NOT ${HPCOMBI_HAVE_X86INTRIN})
  message(FATAL_ERROR "No SSE or AVX compiler intrinsics")
endif()
file(READ ${CMAKE_SOURCE_DIR}/list_intrin.txt hpcombi_intrinsics)
string(REPLACE ";" "|" hpcombi_intrinsics "${hpcombi_intrinsics}")
string(REPLACE "\n" ";" hpcombi_intrinsics "${hpcombi_intrinsics}")
foreach (intrin ${hpcombi_intrinsics})
  string(REPLACE "|" ";" intrin "${intrin}")
  list(GET intrin 0 intrin_name)
  check_cxx_symbol_exists("${intrin_name}" "x86intrin.h"
    "HPCOMBI_HAVE_${intrin_name}")
endforeach()


include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/fallback
  ${PROJECT_BINARY_DIR})

set(benchmark_src bench_epu8.cpp bench_perm16.cpp sort.cpp inverse.cpp)

foreach(f ${benchmark_src})
  get_filename_component(benchName ${f} NAME_WE)
  add_executable (${benchName} ${f})
  target_compile_options(${benchName} PUBLIC -mavx -mtune=native -funroll-loops -flax-vector-conversions)
  target_link_libraries(${benchName} benchmark pthread)
  # install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${benchName}
  #   DESTINATION bin
  #   RENAME ${CMAKE_PROJECT_NAME}-${benchName})
endforeach(f)
