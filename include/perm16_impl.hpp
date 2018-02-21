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
#include "fonctions_gpu.cuh"
#include "power.hpp"
#include <algorithm>
#include <iomanip>
#include <random>

#ifdef HAVE_EXPERIMENTAL_NUMERIC_LCM
#include <experimental/numeric>  // lcm until c++17
#else
#include "fallback/gcdlcm.hpp"  // lcm until c++17
#endif  // HAVE_EXPERIMENTAL_NUMERIC_LCM

#if COMPILE_CUDA==1
extern MemGpu memGpu;
#endif  // USE_CUDA

namespace HPCombi {


// Definitions since previously *only* declared
HPCOMBI_CONSTEXPR size_t Vect16::Size;

/*****************************************************************************/
/** Implementation part for inline functions *********************************/
/*****************************************************************************/

Vect16::Vect16(std::initializer_list<uint8_t> il, uint8_t def) {
  assert(il.size() <= Size);
  std::copy(il.begin(), il.end(), begin());
  auto &a = as_array();
  for (size_t i = il.size(); i < Size; ++i)
    a[i] = def;
}

Vect16 Vect16::random(uint16_t bnd) {
  Vect16 res;
  std::random_device rd;

  std::default_random_engine e1(rd());
  std::uniform_int_distribution<int> uniform_dist(0, bnd - 1);
  for (size_t i = 0; i < Size; i++)
    res.v[i] = uniform_dist(e1);
  return res;
}

// Comparison mode for _mm_cmpestri
#define FIRST_DIFF \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_NEGATIVE_POLARITY)
#define LAST_DIFF (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | \
                        _SIDD_NEGATIVE_POLARITY | _SIDD_MOST_SIGNIFICANT)
#define FIRST_ZERO (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY)
#define LAST_ZERO \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT)
#define FIRST_NON_ZERO \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MASKED_NEGATIVE_POLARITY)
#define LAST_NON_ZERO \
    (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MASKED_NEGATIVE_POLARITY | \
     _SIDD_MOST_SIGNIFICANT)
#define FIND_IN_PERM (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | \
                           _SIDD_UNIT_MASK | _SIDD_NEGATIVE_POLARITY)

inline uint64_t Vect16::first_diff(const Vect16 &b, size_t bound) const {
  return unsigned(_mm_cmpestri(v, bound, b.v, bound, FIRST_DIFF));
}
inline bool Vect16::operator==(const Vect16 &b) const {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) == 0xffff;
  // return first_diff(b) == Size;
}
inline bool Vect16::operator!=(const Vect16 &b) const {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(v, b.v)) != 0xffff;
  // return first_diff(b) != Size;
}
inline bool Vect16::operator<(const Vect16 &b) const {
  uint64_t diff = first_diff(b);
  return (diff < Size) && v[diff] < b[diff];
}
inline char Vect16::less_partial(const Vect16 &b, int k) const {
  uint64_t diff = first_diff(b, k);
  return (diff == Size)
             ? 0
             : static_cast<char>(v[diff]) - static_cast<char>(b[diff]);
}
inline Vect16 Vect16::permuted(const Vect16 &other) const {
  return _mm_shuffle_epi8(v, other);
}


#if COMPILE_CUDA==1
inline Vect16 Vect16::permuted_gpu(const Vect16 &other) const {

  // Simple pointers are needed to cpy to GPU
  float timers[4] = {0, 0, 0, 0};
  Vect16 res;
  // Use of preallocated pinned memory
	// Copy to pinned memory
  for(auto i=0; i<Size; i++){
	  memGpu.h_x8[i] = (*this)[i];
	  memGpu.h_y8[i] = other[i];
  }
  
  shufl_gpu<uint8_t>(memGpu.h_x8, memGpu.h_y8, Size, timers);

  // Copy result from pinned memory
  for(auto i=0; i<Size; i++){
	  res[i] = memGpu.h_x8[i];
  }
  return res;
}

#endif  // USE_CUDA

inline uint8_t Vect16::sum_ref() const {
  uint8_t res = 0;
  for (size_t i = 0; i < Size; i++)
    res += v[i];
  return res;
}

inline uint8_t Vect16::sum4() const { return partial_sums_round()[15]; }

inline uint8_t Vect16::sum3() const {
  Vect16 res = *this;
  res.v += res.permuted(summing_rounds[0]).v;
  res.v += res.permuted(summing_rounds[1]).v;
  res.v += res.permuted(summing_rounds[2]).v;
  return res.v[7] + res.v[15];
}

inline Vect16 Vect16::partial_sums_ref() const {
  Vect16 res{};
  res[0] = v[0];
  for (size_t i = 1; i < Size; i++)
    res[i] = res[i - 1] + v[i];
  return res;
}
inline Vect16 Vect16::partial_sums_round() const {
  Vect16 res = *this;
  for (Vect16 round : summing_rounds)
    res.v += res.permuted(round).v;
  return res;
}

inline Vect16 Vect16::eval16_ref() const {
  Vect16 res;
  for (size_t i = 0; i < Size; i++)
    if (v[i] < Size)
      res[v[i]]++;
  return res;
}

inline Vect16 Vect16::eval16_vect() const {
  Vect16 res, vp = v;
  res.v = -(Perm16::one().v == vp.v);
  for (int i = 1; i < 16; i++) {
    vp = vp.permuted(Perm16::left_cycle());
    res.v -= (Perm16::one().v == vp.v);
  }
  return res;
}

template <char IDX_MODE> inline uint64_t Vect16::search_index(int bound) const {
  return unsigned(_mm_cmpestri(epu8{}, 1, v, bound, IDX_MODE));
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


inline bool Vect16::is_partial_transformation(const size_t k) const {
  uint64_t diff =
      unsigned(_mm_cmpestri(v, Size, Perm16::one(), Size, LAST_DIFF));
  return
    (_mm_movemask_epi8(v+cst_epu8_0x01 <= cst_epu8_0x0F) == 0xffff) &&
    (diff == Size || diff < k);
}

inline bool Vect16::is_transformation(const size_t k) const {
  uint64_t diff =
      unsigned(_mm_cmpestri(v, Size, Perm16::one(), Size, LAST_DIFF));
  return
    (_mm_movemask_epi8(v < cst_epu8_0x0F) == 0xffff) &&
    (diff == Size || diff < k);
}

inline bool Vect16::is_permutation(const size_t k) const {
  uint64_t diff =
      unsigned(_mm_cmpestri(v, Size, Perm16::one(), Size, LAST_DIFF));

  // (forall x in v, x in Perm16::one())  and
  // (forall x in Perm16::one(), x in v)  and
  // (v = Perm16::one()   or  last diff index < Size)
  return _mm_cmpestri(Perm16::one(), Size, v, Size, FIRST_NON_ZERO) == Size &&
         _mm_cmpestri(v, Size, Perm16::one(), Size, FIRST_NON_ZERO) == Size &&
         (diff == Size || diff < k);
}

inline Vect16 Vect16::sorted() const {
  Vect16 res = *this;
  for (auto round : sorting_rounds) {
    Vect16 b = res.permuted(round);

    Vect16 mask = _mm_cmplt_epi8(round, Perm16::one());
    Vect16 minab = _mm_min_epu8(res, b);  // unsigned comparison
    Vect16 maxab = _mm_max_epu8(res, b);  // unsigned comparison

    res = _mm_blendv_epi8(minab, maxab, mask);
  }
  return res;
}

inline Vect16 Vect16::revsorted() const {
  Vect16 res = *this;
  for (auto round : sorting_rounds) {
    Vect16 b = res.permuted(round);

    Vect16 mask = _mm_cmplt_epi8(round, Perm16::one());
    Vect16 minab = _mm_min_epu8(res, b);  // unsigned comparison
    Vect16 maxab = _mm_max_epu8(res, b);  // unsigned comparison

    res = _mm_blendv_epi8(maxab, minab, mask);
  }
  return res;
}

PTransf16::PTransf16(std::initializer_list<uint8_t> il) {
  assert(il.size() <= Size);
  std::copy(il.begin(), il.end(), begin());
  for (size_t i = il.size(); i < Size; ++i)
    v[i] = i;
}

static HPCOMBI_CONSTEXPR
uint8_t hilo_exchng_fun(uint8_t i) { return i < 8 ? i + 8 : i - 8; }
static HPCOMBI_CONSTEXPR epu8 hilo_exchng = make_epu8(hilo_exchng_fun);
static HPCOMBI_CONSTEXPR
uint8_t hilo_mask_fun(uint8_t i) { return i < 8 ? 0x0 : 0xFF; }
static HPCOMBI_CONSTEXPR epu8 hilo_mask = make_epu8(hilo_mask_fun);

inline Transf16::Transf16(uint64_t compressed) {
  epu8 res = _mm_set_epi64x(compressed, compressed);
  v = _mm_blendv_epi8(res & cst_epu8_0x0F, res >> 4, hilo_mask);
}

inline Transf16::operator uint64_t() const {
  Vect16 res = static_cast<epu8>(_mm_slli_epi32(v, 4));
  res = res.permuted(hilo_exchng).v + v;
  return _mm_extract_epi64(res, 0);
}


Perm16 Perm16::random() {
  Perm16 res = one();
  std::random_shuffle(res.begin(), res.end());
  return res;
}

// From Ruskey : Combinatorial Generation page 138
Perm16 Perm16::unrankSJT(int n, int r) {
  int j;
  std::array<int, 16> dir;
  Perm16 res{};
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
  assert(i < vect::Size);
  Perm16 res = one();
  res[i] = i + 1;
  res[i + 1] = i;
  return res;
}

inline Perm16 Perm16::inverse_ref() const {
  Vect16 res;
  for (size_t i = 0; i < Size; ++i)
    res.v[v[i]] = i;
  return res;
}

inline Perm16 Perm16::inverse_arr() const {
  Vect16 res;
  auto &arres = res.as_array();
  auto self = as_array();
  for (size_t i = 0; i < Size; ++i)
    arres[self[i]] = i;
  return res;
}

inline Perm16 Perm16::inverse_sort() const {
  // G++-7 compile this shift by 3 additions.
  // Vect16 res = (v << 4) + one().v;
  // I call directly the shift intrinsic
  Vect16 res = static_cast<epu8>(_mm_slli_epi32(v, 4)) + one().v;
  res = res.sorted().v & cst_epu8_0x0F;
  return res;
}

inline Perm16 Perm16::inverse_find() const {
  Perm16 s = *this;
  Vect16 res;
  res.v = -static_cast<epu8>(_mm_cmpestrm(s.v, 8, one(), 16, FIND_IN_PERM));
  for (Perm16 round : inverting_rounds) {
    s = s * round;
    res.v <<= 1;
    res.v -= static_cast<epu8>(_mm_cmpestrm(s.v, 8, one(), 16, FIND_IN_PERM));
  }
  return res;
}

// We declare PERM16 as a correct Monoid
namespace power_helper {

using Perm16 = Perm16;

template <> struct Monoid<Perm16> {
  static const Perm16 one;
  static Perm16 prod(Perm16 a, Perm16 b) { return a * b; }
};

const Perm16 power_helper::Monoid<Perm16>::one = Perm16::one();
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

inline Vect16 Perm16::lehmer_ref() const {
  Vect16 res;
  for (size_t i = 0; i < Size; i++)
    for (size_t j = i + 1; j < Size; j++)
      if (v[i] > v[j])
        res[i]++;
  return res;
}

inline Vect16 Perm16::lehmer() const {
  Vect16 vsh = *this, res = -one().v;
  for (int i = 1; i < 16; i++) {
    vsh = vsh.permuted(left_shift_ff());
    res.v -= (v >= vsh.v);
  }
  return res;
}

inline uint8_t Perm16::length_ref() const {
  uint8_t res = 0;
  for (size_t i = 0; i < Size; i++)
    for (size_t j = i + 1; j < Size; j++)
      if (v[i] > v[j])
        res++;
  return res;
}
inline uint8_t Perm16::length() const { return lehmer().sum(); }

inline uint8_t Perm16::nb_descent_ref() const {
  uint8_t res = 0;
  for (size_t i = 0; i < Size - 1; i++)
    if (v[i] > v[i + 1])
      res++;
  return res;
}
inline uint8_t Perm16::nb_descent() const {
  Perm16 pdec = permuted(left_shift());
  pdec = (v > pdec.v);
  return _mm_popcnt_u32(_mm_movemask_epi8(pdec));
}

inline uint8_t Perm16::nb_cycles_ref() const {
  std::array<bool, Size> b{};
  uint8_t c = 0;
  for (size_t i = 0; i < Size; i++) {
    if (not b[i]) {
      for (size_t j = i; not b[j]; j = v[j])
        b[j] = true;
      c++;
    }
  }
  return c;
}

inline Vect16 Perm16::cycles_mask_unroll() const {
  Vect16 x0, x1 = one();
  Perm16 p = *this;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p * p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  p = p * p;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p * p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  return x1;
}

inline uint8_t Perm16::nb_cycles_unroll() const {
  Vect16 res = one().v == cycles_mask_unroll().v;
  return _mm_popcnt_u32(_mm_movemask_epi8(res));
}

std::ostream &operator<<(std::ostream &stream, Vect16 const &term) {
  stream << "[" << std::setw(2) << unsigned(term[0]);
  for (unsigned i = 1; i < Vect16::Size; ++i)
    stream << "," << std::setw(2) << unsigned(term[i]);
  stream << "]";
  return stream;
}

// clang-format off

// Sorting network Knuth AoCP3 Fig. 51 p 229.
const std::array<Perm16, 9> Vect16::sorting_rounds = {{
    //     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
    epu8 { 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14},
    epu8 { 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13},
    epu8 { 4,  5,  6,  7,  0,  1,  2,  3, 12, 13, 14, 15,  8,  9, 10, 11},
    epu8 { 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7},
    epu8 { 0,  2,  1, 12,  8, 10,  9, 11,  4,  6,  5,  7,  3, 14, 13, 15},
    epu8 { 0,  4,  8, 10,  1,  9, 12, 13,  2,  5,  3, 14,  6,  7, 11, 15},
    epu8 { 0,  1,  4,  5,  2,  3,  8,  9,  6,  7, 12, 13, 10, 11, 14, 15},
    epu8 { 0,  1,  2,  6,  4,  8,  3, 10,  5, 12,  7, 11,  9, 13, 14, 15},
    epu8 { 0,  1,  2,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 13, 14, 15}
  }};

// Gather at the front numbers with (3-i)-th bit not set.
const std::array<Perm16, 3> Perm16::inverting_rounds = {{
    //     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
    epu8 { 0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15},
    epu8 { 0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15},
    epu8 { 0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15}
  }};


#if defined(FF)
#error FF is defined !
#endif /* FF */
#define FF 0xff

const std::array<epu8, 4> Vect16::summing_rounds = {{
    //      0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
    epu8 { FF,  0, FF,  2, FF,  4, FF,  6, FF,  8, FF, 10, FF, 12, FF, 14},
    epu8 { FF, FF,  1,  1, FF, FF,  5,  5, FF, FF,  9,  9, FF, FF, 13, 13},
    epu8 { FF, FF, FF, FF,  3,  3,  3,  3, FF, FF, FF, FF, 11, 11, 11, 11},
    epu8 { FF, FF, FF, FF, FF, FF, FF, FF,  7,  7,  7,  7,  7,  7,  7,  7}
  }};

#undef FF

// clang-format on

}  // namespace HPCombi
