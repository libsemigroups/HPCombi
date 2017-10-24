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

#include <algorithm>

/*****************************************************************************/
/** Implementation part for inline functions *********************************/
/*****************************************************************************/

inline Vect16::Vect16(std::initializer_list<uint8_t> il) {
  assert(il.size() <= Size);
  std::copy(il.begin(), il.end(), begin());
  for (uint64_t i = il.size(); i < Size; ++i) v[i] = 0;
}

// Comparison mode for _mm_cmpestri
const char FIRST_DIFF = (
  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY);
const char LAST_DIFF = (
  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH |
  _SIDD_NEGATIVE_POLARITY | _SIDD_MOST_SIGNIFICANT);
const char FIRST_ZERO = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY);
const char LAST_ZERO = (
  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT);
const char FIRST_NON_ZERO = (
  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MASKED_NEGATIVE_POLARITY);
const char LAST_NON_ZERO = (
  _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
  _SIDD_MASKED_NEGATIVE_POLARITY | _SIDD_MOST_SIGNIFICANT);
const char FIND_IN_PERM = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
                           _SIDD_UNIT_MASK | _SIDD_NEGATIVE_POLARITY);

static constexpr const epu8 idv =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

inline uint64_t Vect16::first_diff(const Vect16 &b, size_t bound) const {
  return unsigned(_mm_cmpestri (v, bound, b.v, bound, FIRST_DIFF));
}
inline bool Vect16::operator==(const Vect16 &b) const {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) == 0xffff;
  // return first_diff(b) == Size;
}
inline bool Vect16::operator!=(const Vect16 &b) const {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) != 0xffff;
  // return first_diff(b) != Size;
}
inline bool Vect16::operator < (const Vect16 &b) const {
  uint64_t diff = first_diff(b);
  return (diff < Size) && v[diff] < b[diff];
}
inline char Vect16::less_partial(const Vect16 &b, int k) const {
  uint64_t diff = first_diff(b, k);
  return (diff == Size) ? 0 :
    static_cast<char>(v[diff]) - static_cast<char>(b[diff]);
}
inline Vect16 Vect16::permuted(const Vect16 &other) const {
  return _mm_shuffle_epi8(v, other);
}

inline uint8_t Vect16::sum_ref() const {
  uint8_t res = 0;
  for (int i = 0; i < 16; i++) res+=v[i];
  return res;
}

inline uint8_t Vect16::sum() const {
  Vect16 res = *this;
  for (Vect16 round : Vect16::summing_rounds)
    res.v += res.permuted(round).v;
  return _mm_extract_epi8(res, 0);
}

template <char IDX_MODE>
inline uint64_t Vect16::search_index(int bound) const {
  return unsigned(_mm_cmpestri(epu8 {}, 1, v, bound, IDX_MODE));
}

inline uint64_t Vect16::last_non_zero(int bnd) const {
  return search_index<LAST_NON_ZERO>(bnd);
}
inline uint64_t Vect16::first_non_zero(int bnd) const {
  return search_index<FIRST_NON_ZERO>(bnd);
}
inline uint64_t Vect16::last_zero(int bnd) const {
  return search_index<LAST_ZERO>(bnd);
}
inline uint64_t Vect16::first_zero(int bnd) const {
  return search_index<FIRST_ZERO>(bnd);
}
inline bool Vect16::is_permutation(const size_t k) const {
  uint64_t diff = unsigned(_mm_cmpestri(v, Size, idv, Size, LAST_DIFF));

    // (forall x in v, x in idv)  and  (forall x in idv, x in v)  and
    // (v = idv  or  last diff index < Size)
  return
    _mm_cmpestri(idv, Size, v, Size, FIRST_NON_ZERO) == Size &&
    _mm_cmpestri(v, Size, idv, Size, FIRST_NON_ZERO) == Size &&
    (diff == Size || diff < k);
}

static_assert(sizeof(Vect16) == sizeof(Perm16),
              "Vect16 and Perm16 have a different memory layout !");

const uint64_t prime = 0x9e3779b97f4a7bb9;

inline Vect16 Vect16::sorted() const {
  Vect16 res = *this;
  for (Vect16 round : sorting_rounds) {
    Vect16 minab, maxab, blend, mask, b = res.permuted(round);

    mask = _mm_cmplt_epi8(round, Perm16::one());
    minab = _mm_min_epu8(res, b);  // unsigned comparison
    maxab = _mm_max_epu8(res, b);  // unsigned comparison

    res = _mm_blendv_epi8(minab, maxab, mask);
  }
  return res;
}

inline Vect16 Vect16::revsorted() const {
  Vect16 res = *this;
  for (Vect16 round : sorting_rounds) {
    Vect16 minab, maxab, blend, mask, b = res.permuted(round);

    mask = _mm_cmplt_epi8(round, Perm16::one());
    minab = _mm_min_epu8(res, b);  // unsigned comparison
    maxab = _mm_max_epu8(res, b);  // unsigned comparison

    res = _mm_blendv_epi8(maxab, minab, mask);
  }
  return res;
}


inline Perm16::Perm16(std::initializer_list<uint8_t> il) {
  assert(il.size() <= vect::Size);
  std::copy(il.begin(), il.end(), begin());
  for (uint64_t i = il.size(); i < vect::Size; ++i) v[i] = i;
}

inline Perm16 Perm16::inverse_ref() const {
  Perm16 res;
  for (uint64_t i = 0; i < vect::Size; ++i) res.v[v[i]] = i;
  return res;
}

inline Perm16 Perm16::inverse_sort() const {
  Vect16 res = (v << 4) + one().v;
  res = res.sorted().v & 0xf;
  return res;
}

inline Perm16 Perm16::inverse_find() const {
  Perm16 res, s = *this;
  res.v = -epu8(_mm_cmpestrm(s.v, 8, idv, 16, FIND_IN_PERM));
  for (Perm16 round : inverting_rounds) {
    s = s * round;
    res.v <<= 1;
    res.v -= epu8(_mm_cmpestrm(s.v, 8, idv, 16, FIND_IN_PERM));
  }
  return res;
}


#include "power.hpp"

namespace power_helper {

template <> struct Monoid<Perm16> {
  static constexpr const Perm16 one = Perm16::one();
  static Perm16 mult(Perm16 a, Perm16 b) { return a * b; }
};

};  // namespace power_helper


inline Perm16 Perm16::inverse_pow() const {
  Perm16 res;
  Perm16 newpow = pow<8>(*this);
  for (int i=9; i <= 16; i++) {
    Perm16 oldpow = newpow;
    newpow = oldpow * *this;
    res.v = _mm_blendv_epi8(res, oldpow, newpow.v == one().v);
  }
  return res;
}

inline Vect16 Perm16::lehmer_ref() const {
  Vect16 res = {{}};
  for (int i=0; i < 16; i++)
    for (int j=i+1; j < 16; j++)
      if (v[i] > v[j]) res[i]++;
  return res;
}

const Vect16 left_shift =
     { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 255};

inline Vect16 Perm16::lehmer() const {
  Vect16 vsh = *this, res = -Perm16::one().v;
  for (int i=1; i < 16; i++) {
    vsh = vsh.permuted(left_shift);
    res.v -= (v >= vsh.v);
  }
  return res;
}
inline uint8_t Perm16::length_ref() const {
  uint8_t res = 0;
  for (int i=0; i < 16; i++)
    for (int j=i+1; j < 16; j++)
      if (v[i] > v[j]) res++;
  return res;
}
inline uint8_t Perm16::length() const {
    return lehmer().sum();
}
