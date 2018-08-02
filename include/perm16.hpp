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
#include <vector>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <ostream>
#include <x86intrin.h>

#include "epu.hpp"

namespace HPCombi {


// Forward declaration
struct Perm16;
struct PTransf16;
struct Transf16;



/** Test for partial transformation
 * @details
 * @returns whether \c v is a partial transformation.
 * @param v the vector to test
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
inline bool is_partial_transformation(epu8 v, const size_t k = 16);
/** Partial transformation of @f$\{0\dots 15\}@f$
 *
 */
struct PTransf16 {
    epu8 v;

    PTransf16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PTransf16(const epu8 x) : v(x) {}
    PTransf16(std::initializer_list<uint8_t> il);
    operator epu8() const { return v; };

    static HPCOMBI_CONSTEXPR PTransf16 one() { return epu8id; }
    PTransf16 inline operator*(const PTransf16 &p) const {
        return permuted(v, p.v) | (v == Epu8(0xFF));
    }

    bool operator==(const PTransf16 &x) const { return equal(v, x.v); }
    bool operator!=(const PTransf16 &x) const { return not_equal(v, x.v); }
    uint8_t operator[](size_t i) const { return v[i]; }

    uint32_t image() const;
};

/** Test for transformation
 * @details
 * @returns whether \c *this is a transformation.
 * @param v the vector to test
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
inline bool is_transformation(epu8 v, const size_t k = 16);
/** Full transformation of @f$\{0\dots 15\}@f$
 *
 */
struct Transf16 : public PTransf16 {

    Transf16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Transf16(const epu8 x) : PTransf16(x) {}
    Transf16(std::initializer_list<uint8_t> il) : PTransf16(il) {}
    explicit Transf16(uint64_t compressed);

    explicit operator uint64_t() const;

    static HPCOMBI_CONSTEXPR Transf16 one() { return epu8id; }
    Transf16 inline operator*(const Transf16 &p) const { return permuted(v, p.v); }
};


/** Test for permutations
 * @details
 * @returns whether \c *this is a permutation.
 * @param v the vector to test
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
inline bool is_permutation(epu8 v, const size_t k = 16);
/** Permutations of @f$\{0\dots 15\}@f$
 *
 */
struct Perm16 : public Transf16 {

  Perm16() = default;
  HPCOMBI_CONSTEXPR_CONSTRUCTOR Perm16(const epu8 x) : Transf16(x) {}
  Perm16(std::initializer_list<uint8_t> il) : Transf16(il) {}
  explicit Perm16(uint64_t compressed) : Transf16(compressed) {}
  Perm16 inline operator*(const Perm16 &p) const { return permuted(v, p.v); }

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
  static HPCOMBI_CONSTEXPR
  Perm16 one() { return epu8id; }

  inline static Perm16 elementary_transposition(uint64_t i);
  inline static Perm16 random();
  inline static Perm16 unrankSJT(int n, int r);

  inline epu8 lehmer_ref() const;
  inline epu8 lehmer() const;
  inline uint8_t length_ref() const;
  inline uint8_t length() const;

  inline uint8_t nb_descent_ref() const;
  inline uint8_t nb_descent() const;

  inline uint8_t nb_cycles_ref() const;
  inline epu8 cycles_mask_unroll() const;
  inline uint8_t nb_cycles_unroll() const;
  inline uint8_t nb_cycles() const { return nb_cycles_unroll(); }

 private:
  inline static const std::array<Perm16, 3> inverting_rounds();
};

/*****************************************************************************/
/** Memory layout concepts check  ********************************************/
/*****************************************************************************/

static_assert(sizeof(epu8) == sizeof(Perm16),
              "epu8 and Perm16 have a different memory layout !");
static_assert(std::is_trivial<epu8>(), "epu8 is not a trivial class !");
static_assert(std::is_trivial<Perm16>(), "Perm16 is not a trivial class !");

}  // namespace HPCombi

#include "perm16_impl.hpp"

namespace std {

template <> struct hash<HPCombi::PTransf16> {
    inline size_t operator()(const HPCombi::PTransf16 &ar) const {
        return std::hash<HPCombi::epu8>{}(ar.v);
    }
};

template <> struct hash<HPCombi::Transf16> {
    inline size_t operator()(const HPCombi::Transf16 &ar) const {
        return uint64_t(ar);
    }
};

template <> struct hash<HPCombi::Perm16> {
    inline size_t operator()(const HPCombi::Perm16 &ar) const {
        return uint64_t(ar);
    }
};

}  // namespace std

#endif  // HPCOMBI_PERM16_HPP_INCLUDED
