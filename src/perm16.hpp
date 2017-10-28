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

#ifndef PERM16_PERM16_HPP_INCLUDED
#define PERM16_PERM16_HPP_INCLUDED

#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <array>
#include <ostream>
#include <functional>  // less<>
#include <algorithm>


namespace IVMPG {

using epu8 = uint8_t __attribute__ ((vector_size (16)));


struct alignas(16) Vect16 {
  static const constexpr size_t Size = 16;
  epu8 v;

  // Overload the default copy constructor and operator= : 10% speedup
  Vect16() = default;
  constexpr Vect16(const Vect16 &x) : v(x.v) {}
  constexpr Vect16(epu8 x) : v(x) {}
  Vect16(std::initializer_list<uint8_t> il);
  constexpr operator epu8() { return v; }
  constexpr operator const epu8() const { return v; }

  Vect16 & operator=(const Vect16 &x) {v = x.v; return *this;}
  Vect16 & operator=(const epu8 &vv) {v = vv; return *this;}

  constexpr uint8_t operator[](uint64_t i) const { return v[i]; }
  constexpr uint8_t & operator[](uint64_t i) { return v[i]; }

  constexpr std::array<uint8_t, 16> &as_array() {
    return reinterpret_cast<std::array<unsigned char, 16>&>(v); }

  auto begin() { return as_array().begin(); }
  auto end() { return as_array().end(); }

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

  static Vect16 random(uint16_t bnd = 256);

 private:
  static const std::array<Vect16, 9> sorting_rounds;
  static const std::array<Vect16, 4> summing_rounds;
};

std::ostream & operator<<(std::ostream & stream, const Vect16 &term);

struct Perm16 : public Vect16 {
  using vect = Vect16;

  Perm16() = default;  // : Vect16({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) {};
  constexpr Perm16(const vect v) : vect(v) {}
  Perm16(std::initializer_list<uint8_t> il);

  Perm16 operator*(const Perm16&p) const { return permuted(p); }
  Perm16 inverse_ref() const;
  Perm16 inverse_sort() const;
  Perm16 inverse_find() const;
  Perm16 inverse_pow() const;
  Perm16 inverse_cycl() const;
  inline Perm16 inverse() { return inverse_cycl(); }

  // It's not possible to have a static constexpr member of same type as class
  // being defined (see https://stackoverflow.com/questions/11928089/)
  // therefore we chose to have functions.
  static const constexpr Perm16 one() {
    return Vect16(epu8 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  }
  static const constexpr Perm16 left_cycle() {
    return Vect16(epu8 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  }
  static const constexpr Perm16 right_cycle() {
    return Vect16(epu8 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0});
  }

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

#include "perm16_impl.hpp"

}  // namespace IVMPG


namespace std {

template<>
struct hash<IVMPG::Vect16> {
  inline size_t operator () (const IVMPG::Vect16 &ar) const {
    uint64_t v1 = _mm_extract_epi64(ar.v, 1);
    return (v1*IVMPG::prime) >> 52;

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
    __m128 v1v = __m128(v1.v), v2v = __m128(v2.v);
    return v1v[0] == v2v[0] ? v1v[1] < v2v[1] : v1v[0] < v2v[0];
  }
};

}  // namespace std

#endif   // PERM16_PERM16_HPP_INCLUDED
