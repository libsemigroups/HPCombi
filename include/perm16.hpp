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

#ifndef HPCOMBI_PERM16_HPP_INCLUDED
#define HPCOMBI_PERM16_HPP_INCLUDED

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <ostream>
#include <x86intrin.h>

namespace HPCombi {

/// SIMD vector of 16 unsigned bytes
using epu8 = uint8_t __attribute__((vector_size(16)));


///
template <class Function, std::size_t... Indices>
constexpr epu8 make_epu8_helper(Function f, std::index_sequence<Indices...>) {
  return epu8{f(Indices)...};
}

template <class Function> constexpr epu8 make_epu8(Function f) {
  return make_epu8_helper(f, std::make_index_sequence<16>{});
}

// The four following function should be constexpr lambdas writen directly in
// their corresponding methods. However until C++17, constexpr lambda are
// forbidden. So we put them here.

/// The image of i by the identity function
static constexpr uint8_t make_one(uint8_t i) { return i; }
/// The image of i by the left cycle function
static constexpr uint8_t make_left_cycle(uint8_t i) { return (i + 15) % 16; }
/// The image of i by the right cycle function
static constexpr uint8_t make_right_cycle(uint8_t i) { return (i + 1) % 16; }
/// The image of i by a left shift filling the hole with a @p 0xff
static constexpr uint8_t make_left_shift_ff(uint8_t i) {
  return i == 15 ? 0xff : i + 1;
}
/// The image of i by a left shift duplicating the hole
static constexpr uint8_t make_left_shift(uint8_t i) {
  return i == 15 ? 15 : i + 1;
}


// Old Clang doesn't automatically broadcast uint8_t into epu8
// We therefore write there the explicit constants
const epu8 cst_epu8_0x00 = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
const epu8 cst_epu8_0x01 = {0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01};
const epu8 cst_epu8_0x02 = cst_epu8_0x01 + cst_epu8_0x01;
const epu8 cst_epu8_0x04 = cst_epu8_0x02 + cst_epu8_0x02;
const epu8 cst_epu8_0x08 = cst_epu8_0x04 + cst_epu8_0x04;
const epu8 cst_epu8_0x0F = {0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
                            0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};
const epu8 cst_epu8_0xF0 = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
                            0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};
const epu8 cst_epu8_0xFF = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Forward declaration
struct Perm16;
struct PTransf16;
struct Transf16;


/** A class for vector of 16 unsigned bytes
 *
 */
struct alignas(16) Vect16 {
  static const constexpr size_t Size = 16;
  epu8 v;

  Vect16() = default;
  constexpr Vect16(epu8 x) : v(x) {}
  Vect16(std::initializer_list<uint8_t> il, uint8_t def = 0);
  constexpr operator const epu8() const { return v; }

  // Overload the default copy constructor and operator= : 10% speedup
  // But result in Non POD
  // constexpr Vect16(const Vect16 &x) : v(x.v) {}
  // Vect16 & operator=(const Vect16 &x) {v = x.v; return *this;}
  Vect16 &operator=(const Vect16 &) = default;
  Vect16 &operator=(const epu8 &vv) {
    v = vv;
    return *this;
  }

  /** Return self as an array (just a cast)
   *
   *  This is usually faster for algorithm using a lot of indexed acces.
   */
  std::array<uint8_t, 16> &as_array() {
    return reinterpret_cast<std::array<unsigned char, 16> &>(v);
  }
  /** Return self as a const array (just a cast)
   *
   *  This is usually faster for algorithm using a lot of indexed acces.
   */
  const std::array<uint8_t, 16> &as_array() const {
    return reinterpret_cast<const std::array<unsigned char, 16> &>(v);
  }

  // The following two functions are refused by clang++
  // const uint8_t & operator[](uint64_t i) const { return v[i]; }
  // uint8_t & operator[](uint64_t i) { return v[i]; }
  const uint8_t &operator[](uint64_t i) const { return as_array()[i]; }
  uint8_t &operator[](uint64_t i) { return as_array()[i]; }

  auto begin() { return as_array().begin(); }
  auto end() { return as_array().end(); }

  inline bool operator==(const Vect16 &b) const;
  inline bool operator!=(const Vect16 &b) const;
  inline bool operator<(const Vect16 &b) const;
  inline char less_partial(const Vect16 &b, int k) const;
  inline Vect16 permuted(const Vect16 &other) const;
  inline Vect16 sorted() const;
  inline Vect16 revsorted() const;

  inline uint64_t first_diff(const Vect16 &b, size_t bound = Size) const;

  inline uint8_t sum_ref() const;
  inline uint8_t sum4() const;
  inline uint8_t sum3() const;
  inline uint8_t sum() const { return sum3(); }

  inline Vect16 partial_sums_ref() const;
  inline Vect16 partial_sums_round() const;

  template <char IDX_MODE> inline uint64_t search_index(int bound) const;

  inline uint64_t last_non_zero(int bnd = Size) const;
  inline uint64_t first_non_zero(int bnd = Size) const;
  inline uint64_t last_zero(int bnd = Size) const;
  inline uint64_t first_zero(int bnd = Size) const;

  inline Vect16 eval16_ref() const;
  inline Vect16 eval16_vect() const;

  /** Test for partial transformation
   * @details
   * @returns whether \c *this is a partial transformation.
   * @param k the size of \c *this (default 16)
   *
   * Points where the function is undefined are mapped to \c 0xff. If \c *this
   * is a tranformation of @f$0\dots n-1@f$ for @f$n<16@f$, it should be completed
   * to a transformation of @f$0\dots 15@f$ by adding fixed points. That is the
   * values @f$i\geq n@f$ should be mapped to themself.
   * @par Example:
   * The partial tranformation
   * @f$\begin{matrix}0 1 2 3 4 5\\ 2 0 5 . . 4 \end{matrix}@f$
   * is encoded by the array {2,0,5,0xff,0xff,4,6,7,8,9,10,11,12,13,14,15}
   */
  inline bool is_partial_transformation(const size_t k = Size) const;
  /** Test for transformation
   * @details
   * @returns whether \c *this is a transformation.
   * @param k the size of \c *this (default 16)
   *
   * If \c *this is a tranformation of @f$0\dots n-1@f$ for @f$n<16@f$,
   * it should be completed to a transformation of @f$0\dots 15@f$
   * by adding fixed points. That is the values @f$i\geq n@f$ should be
   * mapped to themself.
   * @par Example:
   * The tranformation
   * @f$\begin{matrix}0 1 2 3 4 5\\ 2 0 5 2 1 4 \end{matrix}@f$
   * is encoded by the array {2,0,5,2,1,4,6,7,8,9,10,11,12,13,14,15}
   */
  inline bool is_transformation(const size_t k = Size) const;
  /** Test for permutations
   * @details
   * @returns whether \c *this is a permutation.
   * @param k the size of \c *this (default 16)
   *
   * If \c *this is a permutation of @f$0\dots n-1@f$ for @f$n<16@f$,
   * it should be completed to a permutaition of @f$0\dots 15@f$
   * by adding fixed points. That is the values @f$i\geq n@f$ should be
   * mapped to themself.
   * @par Example:
   * The permutation
   * @f$\begin{matrix}0 1 2 3 4 5\\ 2 0 5 3 1 4 \end{matrix}@f$
   * is encoded by the array {2,0,5,3,1,4,6,7,8,9,10,11,12,13,14,15}
   */
  inline bool is_permutation(const size_t k = Size) const;

  static Vect16 random(uint16_t bnd = 256);

private:
  static const std::array<Perm16, 9> sorting_rounds;
  static const std::array<epu8, 4> summing_rounds;
};

std::ostream &operator<<(std::ostream &stream, const Vect16 &term);


/** Partial transformation of @f$\{0\dots 15\}@f$
 *
 */
struct PTransf16 : public Vect16 {
  using vect = Vect16;

  PTransf16() = default;
  constexpr PTransf16(const vect v) : vect(v) {}
  constexpr PTransf16(const epu8 x) : vect(x) {}
  PTransf16(std::initializer_list<uint8_t> il);

  static constexpr PTransf16 one() { return make_epu8(make_one); }
  PTransf16 inline operator*(const PTransf16 &p) const {
    return permuted(p).v | (v == cst_epu8_0xFF); }
};


/** Full transformation of @f$\{0\dots 15\}@f$
 *
 */
struct Transf16 : public PTransf16 {

  Transf16() = default;
  constexpr Transf16(const vect v) : PTransf16(v) {}
  constexpr Transf16(const epu8 x) : PTransf16(x) {}
  Transf16(std::initializer_list<uint8_t> il) : PTransf16(il) {}
  explicit Transf16(uint64_t compressed);

  explicit operator uint64_t() const;

  static constexpr Transf16 one() { return make_epu8(make_one); }
  Transf16 inline operator*(const Transf16 &p) const { return permuted(p); }
};


/** Permutations of @f$\{0\dots 15\}@f$
 *
 */
struct Perm16 : public Transf16 {

  Perm16() = default;
  // constexpr Perm16() : Vect16(epu8 {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})
  // {};
  constexpr Perm16(const vect v) : Transf16(v) {}
  constexpr Perm16(const epu8 x) : Transf16(x) {}
  Perm16(std::initializer_list<uint8_t> il) : Transf16(il) {}
  explicit Perm16(uint64_t compressed) : Transf16(compressed) {}
  Perm16 inline operator*(const Perm16 &p) const { return permuted(p); }

  /** @class common_inverse
   * @brief The inverse permutation
   * @details
   * @returns the inverse of \c *this
   * @par Example:
   * @code
   * Perm16 x = {0,3,2,4,1,5,6,7,8,9,10,11,12,13,14,15};
   * x.inverse()
   * @endcode
   * Returns {0,4,2,1,3,5,6,7,8,9,10,11,12,13,14,15}
   */
  /** @copydoc common_inverse
   *  @par Algorithm:
   *  Reference @f$O(n)@f$ algorithm using loop and indexed access
   */
  inline Perm16 inverse_ref() const;
  /** @copydoc common_inverse
   *  @par Algorithm:
   *  @f$O(n)@f$ algorithm using reference cast to arrays
   */
  inline Perm16 inverse_arr() const;
  /** @copydoc common_inverse
   *  @par Algorithm:
   *  Insert the identity in the least significant bits and sort using a
   *  sorting network. The number of round of the optimal sorting network is
   *  as far as I know open, therefore, the complexity is unknown.
   */
  inline Perm16 inverse_sort() const;
  /** @copydoc common_inverse
   *  @par Algorithm:
   *  @f$O(\log n)@f$ algorithm using some kind of vectorized dichotomic search.
   */
  inline Perm16 inverse_find() const;
  /** @copydoc common_inverse
   *  @par Algorithm:
   *
   * Raise \e *this to power @f$\text{LCM}(1, 2, ..., n) - 1@f$ so complexity
   * is in @f$O(log (\text{LCM}(1, 2, ..., n) - 1)) = O(n)@f$
   */
  inline Perm16 inverse_pow() const;
  /** @copydoc common_inverse
   *  @par Algorithm:
   *  Compute power from @f$n/2@f$ to @f$n@f$, when @f$\sigma^k(i)=i@f$ then
   *  @f$\sigma^{-1}(i)=\sigma^{k-1}(i)@f$. Complexity @f$O(n)@f$
   */
  inline Perm16 inverse_cycl() const;
  /** @copydoc common_inverse
   *
   *  Frontend method: currently aliased to #inverse_cycl */
  inline Perm16 inverse() { return inverse_cycl(); }

  // It's not possible to have a static constexpr member of same type as class
  // being defined (see https://stackoverflow.com/questions/11928089/)
  // therefore we chose to have functions.
  static constexpr Perm16 one() { return make_epu8(make_one); }
  static constexpr Perm16 left_cycle() { return make_epu8(make_left_cycle); }
  static constexpr Perm16 right_cycle() { return make_epu8(make_right_cycle); }
  static constexpr Perm16 left_shift() { return make_epu8(make_left_shift); }
  static constexpr Perm16 left_shift_ff() {
    return make_epu8(make_left_shift_ff);
  }

  inline static Perm16 elementary_transposition(uint64_t i);
  static Perm16 random();
  static Perm16 unrankSJT(int n, int r);

  inline Vect16 lehmer_ref() const;
  inline Vect16 lehmer() const;
  inline uint8_t length_ref() const;
  inline uint8_t length() const;

  inline uint8_t nb_descent_ref() const;
  inline uint8_t nb_descent() const;

  inline uint8_t nb_cycles_ref() const;
  inline Vect16 cycles_mask_unroll() const;
  inline uint8_t nb_cycles_unroll() const;
  inline uint8_t nb_cycles() const { return nb_cycles_unroll(); }

private:
  static const std::array<Perm16, 3> inverting_rounds;
};

/*****************************************************************************/
/** Memory layout concepts check  ********************************************/
/*****************************************************************************/

static_assert(sizeof(Vect16) == sizeof(Perm16),
              "Vect16 and Perm16 have a different memory layout !");
static_assert(std::is_trivial<Vect16>(), "Vect16 is not a trivial class !");
static_assert(std::is_trivial<Perm16>(), "Perm16 is not a trivial class !");

}  // namespace HPCombi

#include "perm16_impl.hpp"

namespace std {

template <> struct hash<HPCombi::Vect16> {
  inline size_t operator()(const HPCombi::Vect16 &ar) const {
    __int128 v0 = _mm_extract_epi64(ar.v, 0);
    __int128 v1 = _mm_extract_epi64(ar.v, 1);
    return ((v1 * HPCombi::prime + v0) * HPCombi::prime) >> 64;
  }
};

template <> struct hash<HPCombi::PTransf16> {
  inline size_t operator()(const HPCombi::PTransf16 &ar) const {
    return hash<HPCombi::Vect16>()(ar);
  }
};

template <> struct hash<HPCombi::Transf16> {
  inline size_t operator()(const HPCombi::Transf16 &ar) const {
    return hash<HPCombi::Vect16>()(ar);
  }
};

template <> struct hash<HPCombi::Perm16> {
  inline size_t operator()(const HPCombi::Perm16 &ar) const {
    return hash<HPCombi::Vect16>()(ar);
  }
};


template <> struct less<HPCombi::Vect16> {
  // WARNING: due to endianess this is not lexicographic comparison,
  //          but we don't care when using in std::set.
  // 10% faster than calling the lexicographic comparison operator !
  inline size_t operator()(const HPCombi::Vect16 &v1,
                           const HPCombi::Vect16 &v2) const {
    __m128 v1v = __m128(v1.v), v2v = __m128(v2.v);
    return v1v[0] == v2v[0] ? v1v[1] < v2v[1] : v1v[0] < v2v[0];
  }
};

}  // namespace std

#endif  // HPCOMBI_PERM16_HPP_INCLUDED
