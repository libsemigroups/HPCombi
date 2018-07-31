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

#include "power.hpp"
#include <algorithm>
#include <iomanip>
#include <random>

#ifdef HAVE_EXPERIMENTAL_NUMERIC_LCM
#include <experimental/numeric>  // lcm until c++17
#else
#include "gcdlcm.hpp"  // lcm until c++17
#endif  // HAVE_EXPERIMENTAL_NUMERIC_LCM

namespace HPCombi {

/*****************************************************************************/
/** Implementation part for inline functions *********************************/
/*****************************************************************************/

#define FIND_IN_PERM (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | \
                           _SIDD_UNIT_MASK | _SIDD_NEGATIVE_POLARITY)
#define FIND_IN_PERM_COMPL (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | \
                           _SIDD_UNIT_MASK)

inline bool is_partial_transformation(epu8 v, const size_t k) {
    uint64_t diff = last_diff(v, epu8id, 16);
    return
        (_mm_movemask_epi8(v+Epu8(1) <= Epu8(0x10)) == 0xffff) &&
        (diff == 16 || diff < k);
}

inline bool is_transformation(epu8 v, const size_t k) {
    uint64_t diff = last_diff(v, epu8id, 16);
    return
        (_mm_movemask_epi8(v < Epu8(0x10)) == 0xffff) &&
        (diff == 16 || diff < k);
}

inline bool is_permutation(epu8 v, const size_t k) {
    uint64_t diff = last_diff(v, epu8id, 16);
    // (forall x in v, x in Perm16::one())  and
    // (forall x in Perm16::one(), x in v)  and
    // (v = Perm16::one()   or  last diff index < 16)
    return
        _mm_cmpestri(Perm16::one(), 16, v, 16, FIRST_NON_ZERO) == 16 &&
        _mm_cmpestri(v, 16, Perm16::one(), 16, FIRST_NON_ZERO) == 16 &&
        (diff == 16 || diff < k);
}

inline PTransf16::PTransf16(std::initializer_list<uint8_t> il) {
    assert(il.size() <= 16);
    std::copy(il.begin(), il.end(), as_array(v).begin());
    for (size_t i = il.size(); i < 16; ++i)
        v[i] = i;
}

inline uint32_t PTransf16::image() const {
  return _mm_movemask_epi8(_mm_cmpestrm(v, 16, one().v, 16, FIND_IN_PERM_COMPL));
}

static HPCOMBI_CONSTEXPR
uint8_t hilo_exchng_fun(uint8_t i) { return i < 8 ? i + 8 : i - 8; }
static HPCOMBI_CONSTEXPR epu8 hilo_exchng = Epu8(hilo_exchng_fun);
static HPCOMBI_CONSTEXPR
uint8_t hilo_mask_fun(uint8_t i) { return i < 8 ? 0x0 : 0xFF; }
static HPCOMBI_CONSTEXPR epu8 hilo_mask = Epu8(hilo_mask_fun);

inline Transf16::Transf16(uint64_t compressed) {
  epu8 res = _mm_set_epi64x(compressed, compressed);
  v = _mm_blendv_epi8(res & Epu8(0x0F), res >> 4, hilo_mask);
}

inline Transf16::operator uint64_t() const {
  epu8 res = static_cast<epu8>(_mm_slli_epi32(v, 4));
  res = permuted(res, hilo_exchng) + v;
  return _mm_extract_epi64(res, 0);
}


inline Perm16 Perm16::random() {
  Perm16 res = one();
  auto ar = as_array(res);
  std::random_shuffle(ar.begin(), ar.end());
  return res;
}

// From Ruskey : Combinatorial Generation page 138
Perm16 Perm16::unrankSJT(int n, int r) {
  int j;
  std::array<int, 16> dir;
  epu8 res{};
  for (j = 0; j < n; ++j)
    res[j] = 0xFF;
  for (j = n - 1; j >= 0; --j) {
    int k, rem, c;
    rem = r % (j + 1);
    r = r / (j + 1);
    if ((r & 1) != 0) {
      k = -1;
      dir[j] = +1;
    } else {
      k = n;
      dir[j] = -1;
    }
    c = -1;
    do {
      k = k + dir[j];
      if (res[k] == 0xFF)
        ++c;
    } while (c < rem);
    res[k] = j;
  }
  return res;
}

inline Perm16 Perm16::elementary_transposition(uint64_t i) {
  assert(i < vect::16);
  epu8 res = one();
  res[i] = i + 1;
  res[i + 1] = i;
  return res;
}

inline Perm16 Perm16::inverse_ref() const {
  epu8 res;
  for (size_t i = 0; i < 16; ++i)
    res[v[i]] = i;
  return res;
}

inline Perm16 Perm16::inverse_arr() const {
  epu8 res;
  auto &arres = as_array(res);
  auto self = as_array(v);
  for (size_t i = 0; i < 16; ++i)
    arres[self[i]] = i;
  return res;
}

inline Perm16 Perm16::inverse_sort() const {
  // G++-7 compile this shift by 3 additions.
  // epu8 res = (v << 4) + one().v;
  // I call directly the shift intrinsic
  epu8 res = static_cast<epu8>(_mm_slli_epi32(v, 4)) + one().v;
  res = sorted(res) & Epu8(0x0F);
  return res;
}

// Gather at the front numbers with (3-i)-th bit not set.
const std::array<Perm16, 3> Perm16::inverting_rounds() {
  static std::array<Perm16, 3> res {{
    //     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
    epu8 { 0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15},
    epu8 { 0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15},
    epu8 { 0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15}
  }};
  return res;
}

inline Perm16 Perm16::inverse_find() const {
  Perm16 s = *this;
  epu8 res;
  res = -static_cast<epu8>(_mm_cmpestrm(s.v, 8, one(), 16, FIND_IN_PERM));
  for (Perm16 round : inverting_rounds()) {
    s = s * round;
    res <<= 1;
    res -= static_cast<epu8>(_mm_cmpestrm(s.v, 8, one(), 16, FIND_IN_PERM));
  }
  return res;
}

// We declare PERM16 as a correct Monoid
namespace power_helper {

using Perm16 = Perm16;

template <> struct Monoid<Perm16> {
  static const Perm16 one() { return Perm16::one(); }
  static Perm16 prod(Perm16 a, Perm16 b) { return a * b; }
};

}  // namespace power_helper

inline Perm16 Perm16::inverse_cycl() const {
  Perm16 res = one();
  Perm16 newpow = pow<8>(*this);
  for (int i = 9; i <= 16; i++) {
    Perm16 oldpow = newpow;
    newpow = oldpow * *this;
    res.v = _mm_blendv_epi8(res, oldpow, newpow.v == one().v);
  }
  return res;
}

static constexpr unsigned lcm_range(unsigned n) {
#if __cplusplus <= 201103L
  return n == 1 ? 1 : std::experimental::lcm(lcm_range(n-1), n);
#else
  unsigned res = 1;
  for (unsigned i = 1; i <= n; ++i)
    res = std::experimental::lcm(res, i);
  return res;
#endif
}

inline Perm16 Perm16::inverse_pow() const {
  return pow<lcm_range(16) - 1>(*this);
}

inline epu8 Perm16::lehmer_ref() const {
  epu8 res;
  for (size_t i = 0; i < 16; i++)
    for (size_t j = i + 1; j < 16; j++)
      if (v[i] > v[j])
        res[i]++;
  return res;
}

inline epu8 Perm16::lehmer() const {
  epu8 vsh = *this, res = -one().v;
  for (int i = 1; i < 16; i++) {
      vsh = shifted_left(vsh);
      res -= (v >= vsh);
  }
  return res;
}

inline uint8_t Perm16::length_ref() const {
    uint8_t res = 0;
    for (size_t i = 0; i < 16; i++)
        for (size_t j = i + 1; j < 16; j++)
            if (v[i] > v[j])
                res++;
  return res;
}
inline uint8_t Perm16::length() const { return horiz_sum(lehmer()); }

inline uint8_t Perm16::nb_descent_ref() const {
    uint8_t res = 0;
    for (size_t i = 0; i < 16 - 1; i++)
        if (v[i] > v[i + 1]) res++;
    return res;
}
inline uint8_t Perm16::nb_descent() const {
    return _mm_popcnt_u32(_mm_movemask_epi8(v > shifted_left(v)));
}

inline uint8_t Perm16::nb_cycles_ref() const {
    std::array<bool, 16> b{};
    uint8_t c = 0;
    for (size_t i = 0; i < 16; i++) {
        if (not b[i]) {
            for (size_t j = i; not b[j]; j = v[j])
                b[j] = true;
            c++;
        }
    }
    return c;
}

inline epu8 Perm16::cycles_mask_unroll() const {
  epu8 x0, x1 = one();
  Perm16 p = *this;
  x0 = _mm_min_epi8(x1, permuted(x1, p));
  p = p * p;
  x1 = _mm_min_epi8(x0, permuted(x0, p));
  p = p * p;
  x0 = _mm_min_epi8(x1, permuted(x1, p));
  p = p * p;
  x1 = _mm_min_epi8(x0, permuted(x0, p));
  return x1;
}

inline uint8_t Perm16::nb_cycles_unroll() const {
    epu8 res = (epu8id == cycles_mask_unroll());
    return _mm_popcnt_u32(_mm_movemask_epi8(res));
}


}  // namespace HPCombi
