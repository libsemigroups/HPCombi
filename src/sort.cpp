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


// Sorting network Knuth AoCP3 Fig. 51 p 229.
constexpr const array<epu8, 9> rounds =
    //   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
    {{ { 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14},
       { 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13},
       { 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11},
       { 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7},
       { 0,  2,  1, 12,  8, 10,  9, 11,  4,  6,  5,  7,  3, 14, 13, 15},
       { 0,  4,  8, 10,  1,  9, 12, 13,  2,  5,  3, 14,  6,  7, 11, 15},
       { 0,  1,  4,  5,  2,  3,  8,  9,  6,  7, 12, 13, 10, 11, 14, 15},
       { 0,  1,  2,  6,  4,  8,  3, 10,  5, 12,  7, 11,  9, 13, 14, 15},
       { 0,  1,  2,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 13, 14, 15}
       }};

Vect16 sort(Vect16 a) {
  for (Perm16 round : rounds) {
    Vect16 minab, maxab, mask, b = a.permuted(round);

    mask = _mm_cmplt_epi8(round, Perm16::one());
    minab = _mm_min_epi8(a, b);
    maxab = _mm_max_epi8(a, b);

    a = _mm_blendv_epi8(minab, maxab, mask);
  }
  return a;
}

struct RoundsMask {
  // commented out due to a bug in gcc
  /* constexpr */ RoundsMask() : arr() {
    for (unsigned i = 0; i < rounds.size(); ++i)
      arr[i] = rounds[i] < Perm16::one().v;
  }
  epu8 arr[rounds.size()];
};

const auto rounds_mask = RoundsMask();

Vect16 sort_pair(Vect16 a) {
  for (unsigned i = 0; i < rounds.size(); ++i) {
    Vect16 minab, maxab, b = a.permuted(rounds[i]);
    minab = _mm_min_epi8(a, b);
    maxab = _mm_max_epi8(a, b);
    a = _mm_blendv_epi8(minab, maxab, rounds_mask.arr[i]);
  }
  return a;
}


Perm16 insertion_sort(Perm16 a) {
  for (int i = 0; i < 16; i++)
    for (int j = i; j > 0 && a[j] < a[j-1]; j--)
      std::swap(a[j], a[j-1]);
  return a;
}

Perm16 radix_sort(Perm16 a) {
  Vect16 stat = {}, res;
  for (int i = 0; i < 16; i++) stat[a[i]]++;
  int c = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < stat[i]; j++) res[c++]=i;
  return res;
}

int main() {
  // Perm16 a = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};

  for (Perm16 round : rounds) {
    assert(round.is_permutation());
    assert(round*round == Perm16::one());
  }

  auto vrand = rand_perms(1000);
  int rep = 10000;
  cout << "Std lib: ";
  double reftime = timethat([vrand]() {
      for (Perm16 v : vrand) {
        std::sort(v.begin(), v.end());
        assert(v == Perm16::one());
      }
    }, rep);
  cout << "Method : ";
  timethat([vrand]() {
      for (Perm16 v : vrand) assert(v.sorted() == Perm16::one());
    }, rep, reftime);
  cout << "Funct  : ";
  timethat([vrand]() {
      for (Perm16 v : vrand) assert(sort(v) == Perm16::one());
    }, rep, reftime);
  cout << "Pair  : ";
  timethat([vrand]() {
      for (Perm16 v : vrand) assert(sort_pair(v) == Perm16::one());
    }, rep, reftime);
  cout << "Insert : ";
  timethat([vrand]() {
      for (Perm16 v : vrand) assert(insertion_sort(v) == Perm16::one());
    }, rep, reftime);
  cout << "Radix16: ";
  timethat([vrand]() {
      for (Perm16 v : vrand) assert(radix_sort(v) == Perm16::one());
    }, rep, reftime);

  return EXIT_SUCCESS;
}
