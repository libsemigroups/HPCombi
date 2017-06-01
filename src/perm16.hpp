//****************************************************************************//
//       Copyright (C) 2014 Florent Hivert <Florent.Hivert@lri.fr>,           //
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

#ifndef PERM16_HPP_
#define PERM16_HPP_

#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <algorithm>
#include <array>
#include <ostream>


namespace IVMPG {

using vect16 = std::array<uint8_t, 16>;
using epi8 = uint8_t __attribute__ ((vector_size (16)));


struct alignas(16) Vect16 {
  static const constexpr size_t Size = 16;

  union {
    vect16 p;
    __m128i v;
    epi8 v8;
  };

  // Overload the default copy constructor and operator= : 10% speedup
  Vect16() = default;
  Vect16(const Vect16 &x) { v = x.v; }
  Vect16(std::initializer_list<uint8_t> il);
  Vect16(__m128i x) { v = x; }
  Vect16(epi8 x) { v8 = x; }
  operator __m128i() { return v; }
  operator const __m128i() const { return v; }

  Vect16 & operator=(const Vect16 &x) {v = x.v; return *this;}
  Vect16 & operator=(const __m128i &vv) {v = vv; return *this;}

  uint8_t operator[](uint64_t i) const { return p[i]; }
  uint8_t & operator[](uint64_t i) { return p[i]; }

  uint64_t first_diff(const Vect16 &b, size_t bound = Size) const;

  bool operator==(const Vect16 &b) const;
  bool operator!=(const Vect16 &b) const;
  bool operator<(const Vect16 &b) const;
  char less_partial(const Vect16 &b, int k) const;
  Vect16 permuted(const Vect16 &other) const;
  Vect16 sorted() const;
  Vect16 revsorted() const;
  uint8_t sum_ref() const;
  uint8_t sum() const;

  template <char IDX_MODE> uint64_t search_index(int bound) const;

  uint64_t last_non_zero(int bnd = Size) const;
  uint64_t first_non_zero(int bnd = Size) const;
  uint64_t last_zero(int bnd = Size) const;
  uint64_t first_zero(int bnd = Size) const;

  bool is_permutation(const size_t k = Size) const;

  static Vect16 random(uint16_t m = 256);

 private:
  static const std::array<Vect16, 9> sorting_rounds;
  static const std::array<Vect16, 4> summing_rounds;
};

std::ostream & operator<<(std::ostream & stream, const Vect16 &term);

struct Perm16 : public Vect16 {
  using vect = Vect16;

  Perm16() = default;  // : Vect16({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) {};
  Perm16(const vect v) : vect(v) {}
  Perm16(std::initializer_list<uint8_t> il);

  Perm16 operator*(const Perm16&p) const { return permuted(p); }
  Perm16 inverse_ref() const;
  Perm16 inverse_sort() const;
  Perm16 inverse() const;

  static const Perm16 one;
  static const Perm16 left_cycle;
  static const Perm16 right_cycle;

  static Perm16 elementary_transposition(uint64_t i);
  static Perm16 random();
  static Perm16 unrankSJT(int n, int r);

  Vect16 lehmer_ref() const;
  Vect16 lehmer() const;
  uint8_t length_ref() const;
  uint8_t length() const;

 private:
  static const std::array<Perm16, 3> inverting_rounds;
};

#include"perm16.impl"

}  // namespace IVMPG


namespace std {

template<>
struct hash<IVMPG::Vect16> {
  inline size_t operator () (const IVMPG::Vect16 &ar) const {
    uint64_t v1 = _mm_extract_epi64(ar.v, 1);
    return (v1*IVMPG::prime) >> 54;

    // Timing for a 1024 hash table with SET_STATISTIC defined
    //////////////////////////////////////////////////////////
    //                                          1 proc   8 proc  collision retry
    // return ((v1*prime + v0)*prime) >> 52;   // 7.39027  1.68039    0.0783%
    // return ((v1 + (v0 << 4))*prime) >> 52;  // 7.67103  1.69188    0.0877%
    // return ((v1 + v0 )*prime) >> 52;        // 7.25443  1.63157    0.267%
    // return (v1*prime) >> 52;                // 7.15018  1.61709    2.16%
    // return 0;                               // 8.0689   2.09339  159.%

    // Indexing acces is always slower.
    // return (ar.v[1]*prime) >> 52;              // 1.68217
  }
};

template<>
struct less<IVMPG::Vect16> {
  // WARNING: due to endianess this is not lexicographic comparison,
  //          but we don't care when using in std::set.
  // 10% faster than calling the lexicographic comparison operator !
  inline size_t operator() (const IVMPG::Vect16 &v1,
                            const IVMPG::Vect16 &v2) const {
    return v1.v[0] == v2.v[0] ? v1.v[1] < v2.v[1] : v1.v[0] < v2.v[0];
  }
};

}  // namespace std

#endif   // PERM16_HPP_
