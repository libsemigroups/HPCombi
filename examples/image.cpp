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

#include "perm16.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <iomanip>
#include <iostream>
#include <utility> // pair
#include <string>
#include <tuple>
#include <vector>
#include <x86intrin.h>
#include "timer.h"

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << '[';
  if (!v.empty()) {
    auto i = v.begin();
    for (; i != --v.end(); ++i)
      out << std::setw(2) << *i << ",";
    out << std::setw(2) << *i;
  }
  out << "]";
  return out;
}

using namespace std;
using namespace HPCombi;

// James favourite

const Transf16 a1 {1, 7, 2, 6, 0, 4, 1, 5};
const Transf16 a2 {2, 4, 6, 1, 4, 5, 2, 7};
const Transf16 a3 {3, 0, 7, 2, 4, 6, 2, 4};
const Transf16 a4 {3, 2, 3, 4, 5, 3, 0, 1};
const Transf16 a5 {4, 3, 7, 7, 4, 5, 0, 4};
const Transf16 a6 {5, 6, 3, 0, 3, 0, 5, 1};
const Transf16 a7 {6, 0, 1, 1, 1, 6, 3, 4};
const Transf16 a8 {7, 7, 4, 0, 6, 4, 1, 7};
const array<Transf16, 8> gens{a1,a2,a3,a4,a5,a6,a7,a8};
//const vector<Transf16> gens{{a1, a2}};

// std::array<std::pair<uint16_t, uint16_t>, 65536> res {};
std::array<std::tuple<uint16_t, uint16_t,
                      std::array<uint16_t, gens.size()>>, 65536> res;

int main() {
  int lg = 0;
  int total = 0;

  vector<Transf16> todo, newtodo;
  // res[Transf16::one().image()] = make_tuple(0xFFFF, 0xFFFF, {});
  get<0>(res[Transf16::one().image()]) = 0xFFFF;
  get<1>(res[Transf16::one().image()]) = 0xFFFF;
  cout << "start" << endl;

  libsemigroups::Timer t;
  todo.push_back(Transf16::one());
  while (todo.size()) {
    newtodo.clear();
    lg++;
    for (auto v : todo) {
      total++;
      uint32_t vim = v.image();
      for (uint8_t i = 0; i < gens.size(); i++) {
        Transf16 el = gens[i] * v;
        uint32_t im = el.image();
        get<2>(res[vim])[i] = im;
        if (get<0>(res[im]) == 0) {
          // cout << el.sorted() << endl;
          newtodo.push_back(el);
          get<0>(res[im]) = gens[i].image();
          get<1>(res[im]) = i;
        }
      }
    }
    swap(todo, newtodo);
    // cout << lg << ", todo = " << todo.size() << ", total = " << total << endl;
  }
  cout << t << endl;
  cout << "lg = " << lg << ", total = " << total << endl;
  exit(0);
}
