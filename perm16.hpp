/******************************************************************************/
/*       Copyright (C) 2014 Florent Hivert <Florent.Hivert@lri.fr>,           */
/*                                                                            */
/*  Distributed under the terms of the GNU General Public License (GPL)       */
/*                                                                            */
/*    This code is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       */
/*   General Public License for more details.                                 */
/*                                                                            */
/*  The full text of the GPL is available at:                                 */
/*                                                                            */
/*                  http://www.gnu.org/licenses/                              */
/******************************************************************************/

#ifndef _PERM16_HPP
#define _PERM16_HPP

#include <cstdint>
#include <x86intrin.h>
#include <array>
#include <cassert>
#include <ostream>

namespace IVMPG {

using vect16 = std::array<uint8_t, 16> ;
using epi8 = uint8_t __attribute__ ((vector_size (16)));

// Comparison mode for _mm_cmpestri
const char FIRST_DIFF = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY);
const char LAST_DIFF = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH |
			_SIDD_NEGATIVE_POLARITY | _SIDD_MOST_SIGNIFICANT );
const char FIRST_ZERO = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY );
const char LAST_ZERO = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT);
const char FIRST_NON_ZERO = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
			     _SIDD_MASKED_NEGATIVE_POLARITY);
const char LAST_NON_ZERO = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
			    _SIDD_MASKED_NEGATIVE_POLARITY | _SIDD_MOST_SIGNIFICANT);


struct alignas(16) Vect16
{

  static const constexpr size_t Size = 16;

  union {
    vect16 p;
    __m128i v;
    epi8 v8;
  };

  // Overload the default copy constructor and operator= : 10% speedup
  Vect16() = default;
  Vect16(const Vect16 &x) { v = x.v; }
  Vect16(std::initializer_list<uint8_t> il) {
    assert (il.size() <= Size);
    std::copy(il.begin(), il.end(), this->p.begin());
    for (uint64_t i = il.size(); i<Size; i++) this->p[i] = 0;
  }

  Vect16(__m128i x) { v = x; }
  operator __m128i() { return v; }
  operator const __m128i() const { return v; }

  Vect16 &operator =(const Vect16 &x) {v = x.v; return *this;}
  Vect16 &operator =(const __m128i &vv) {v = vv; return *this;}

  uint8_t operator[](uint64_t i) const { return p[i]; }
  uint8_t &operator[](uint64_t i) { return p[i]; }

  uint64_t first_diff(const Vect16 &b, size_t bound = Size) const {
    return unsigned(_mm_cmpestri (v, bound, b.v, bound, FIRST_DIFF));
  }

  bool operator==(const Vect16 &b) const {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) == 0xffff;
    // return first_diff(b) == Size;
  }
  bool operator!=(const Vect16 &b) const {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) != 0xffff;
    // return first_diff(b) != Size;
  }

  bool operator < (const Vect16 &b) const {
    uint64_t diff = first_diff(b);
    return (diff < Size) and p[diff] < b[diff];
  }

  char less_partial(const Vect16 &b, int k) const {
    uint64_t diff = first_diff(b, k);
    return (diff == Size) ? 0 : char(p[diff]) - char(b[diff]);
  }


  Vect16 permuted(const Vect16 &other) const {
    return _mm_shuffle_epi8(v, other);
  };

  template <char IDX_MODE>
  uint64_t search_index(int bound) const {
    const __m128i zero {0, 0};
    return unsigned(_mm_cmpestri(zero, 1, v, bound, IDX_MODE));
  }

  uint64_t last_non_zero(int bnd=Size) const { return search_index<LAST_NON_ZERO>(bnd); }
  uint64_t first_non_zero(int bnd=Size) const { return search_index<FIRST_NON_ZERO>(bnd); }
  uint64_t last_zero(int bnd=Size) const { return search_index<LAST_ZERO>(bnd); }
  uint64_t first_zero(int bnd=Size) const { return search_index<FIRST_ZERO>(bnd); }

  bool is_permutation(const size_t k = Size) const {
    constexpr const __m128i idv = __m128i(epi8 {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    uint64_t diff = unsigned(_mm_cmpestri(v, Size, idv, Size, LAST_DIFF));
    return
      _mm_cmpestri(idv, Size, v, Size, FIRST_NON_ZERO) == Size and // all(x in idv for x in v)
      _mm_cmpestri(v, Size, idv, Size, FIRST_NON_ZERO) == Size and // all(x in v for x in idv)
      (diff == Size or diff < k);     // v = idv    or    last diff index < Size
  }

};

std::ostream & operator<<(std::ostream & stream, const Vect16 &term);

struct Perm16 : public Vect16 {

  using vect = Vect16;

  Perm16() {}; // : Vect16({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}) {};
  Perm16(std::initializer_list<uint8_t> il) {
    assert (il.size() <= vect::Size);
    std::copy(il.begin(), il.end(), this->p.begin());
    for (uint64_t i = il.size(); i<vect::Size; i++) this->p[i] = i;
  }
  Perm16 operator*(const Perm16&p) const { return permuted(p); }

  static const Perm16 one;
  static Perm16 elementary_transposition(uint64_t i) {
    assert (i < vect::Size);
    Perm16 res {}; res[i]=i+1; res[i+1]=i; return res; }

  Perm16(const vect v) : vect(v) {};

};

const Perm16 Perm16::one = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

static_assert(sizeof(Vect16) == sizeof(Perm16),
	      "Vect16 and Perm16 have a different memory layout !");

const uint64_t prime = 0x9e3779b97f4a7bb9;

} // namespace IVMPG


namespace std {

  template<>
  struct hash<IVMPG::Vect16> {
    size_t operator () (const IVMPG::Vect16 &ar) const {
      uint64_t v1 = _mm_extract_epi64(ar.v, 1);
      return (v1*IVMPG::prime) >> 54;

      // Timing for a 1024 hash table with SET_STATISTIC defined
      //////////////////////////////////////////////////////////
      //                                             1 proc   8 proc  collision retry
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
    size_t operator () (const IVMPG::Vect16 &v1, const IVMPG::Vect16 &v2) const {
      return v1.v[0] == v2.v[0] ? v1.v[1] < v2.v[1] : v1.v[0] < v2.v[0];
    }

  };
}


#endif

