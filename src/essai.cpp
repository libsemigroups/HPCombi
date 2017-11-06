//****************************************************************************//
//       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           //
//                                                                            //
//  Distributed under the terms of the GNU General Public License (GPL)       //
//                                                                            //
//    This code is distributed in the hope that it will be useful,            //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of          //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       //
//   General Public License for more details.                                 //
//                                                                            //
//  The full text of the GPL is available at:                                 //
//                                                                            //
//                  http://www.gnu.org/licenses/                              //
//****************************************************************************//

#include <x86intrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>

#include "perm16.hpp"
#include "testtools.hpp"

using namespace std;
using namespace std::chrono;
using namespace IVMPG;

template<class Function, std::size_t... Indices>
constexpr epu8 make_epu8_helper(Function f, std::index_sequence<Indices...>) {
    return epu8 { f(Indices)... };
}

template<class Function>
constexpr epu8 make_epu8(Function f) {
    return make_epu8_helper(f, std::make_index_sequence<16>{});
}

constexpr uint8_t fun(uint8_t x) { return x * x; }
constexpr uint8_t id(uint8_t x) { return x; }

int main() {
  std::srand(std::time(0));

  constexpr Perm16 one = make_epu8(id);
  constexpr Perm16 p = make_epu8(fun);
  assert(&p[0] == &(p.as_array()[0]));

  cout << one << endl;
  cout << p << endl << endl;
  cout << int(p.length()) << endl;
  cout << int(p.length_ref()) << endl;
  return EXIT_SUCCESS;
}
