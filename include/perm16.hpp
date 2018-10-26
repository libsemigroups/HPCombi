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
#include <vector>
#include <x86intrin.h>

#include "epu.hpp"
#include "vect16.hpp"

namespace HPCombi {

// Forward declaration
struct Perm16;
struct PTransf16;
struct Transf16;

/** Partial transformation of @f$\{0\dots 15\}@f$
 *
 */
struct alignas(16) PTransf16 : public Vect16 {

    static constexpr size_t Size() { return 16; };

    using vect = HPCombi::Vect16;
    using array = TPUBuild<epu8>::array;

    PTransf16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PTransf16(const PTransf16 &v) = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PTransf16(const vect v) : Vect16(v) {}
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PTransf16(const epu8 x) : Vect16(x) {}
    PTransf16(std::vector<uint8_t> dom, std::vector<uint8_t> rng,
              size_t = 0 /* unused */);
    PTransf16(std::initializer_list<uint8_t> il);

    PTransf16 &operator=(const PTransf16 &) = default;
    PTransf16 &operator=(const epu8 &vv) {
        v = vv;
        return *this;
    }

    bool operator==(const PTransf16 &x) const { return equal(v, x.v); }
    bool operator!=(const PTransf16 &x) const { return not_equal(v, x.v); }

    array &as_array() { return HPCombi::as_array(v); }
    const array &as_array() const { return HPCombi::as_array(v); }

    const uint8_t &operator[](uint64_t i) const { return as_array()[i]; }
    uint8_t &operator[](uint64_t i) { return as_array()[i]; }

    static HPCOMBI_CONSTEXPR PTransf16 one() { return epu8id; }
    PTransf16 inline operator*(const PTransf16 &p) const {
        return HPCombi::permuted(v, p.v) | (p.v == Epu8(0xFF));
    }

    epu8 image_mask(bool complement=false) const;
    epu8 domain_mask(bool complement=false) const;
    PTransf16 right_one() const;
    PTransf16 left_one() const;

    uint32_t rank_ref() const;
    uint32_t rank() const;
};

/** Full transformation of @f$\{0\dots 15\}@f$
 *
 */
struct Transf16 : public PTransf16 {

    Transf16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Transf16(const Transf16 &v) = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Transf16(const vect v) : PTransf16(v) {}
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Transf16(const epu8 x) : PTransf16(x) {}
    Transf16(std::initializer_list<uint8_t> il) : PTransf16(il) {}
    explicit Transf16(uint64_t compressed);
    Transf16 &operator=(const Transf16 &) = default;

    explicit operator uint64_t() const;

    static HPCOMBI_CONSTEXPR Transf16 one() { return epu8id; }
    Transf16 inline operator*(const Transf16 &p) const {
        return HPCombi::permuted(v, p.v);
    }
};

/** Partial permutationof @f$\{0\dots 15\}@f$
 *
 */
struct PPerm16 : public PTransf16 {

    PPerm16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PPerm16(const PPerm16 &v) = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PPerm16(const vect v) : PTransf16(v) {}
    HPCOMBI_CONSTEXPR_CONSTRUCTOR PPerm16(const epu8 x) : PTransf16(x) {}
    PPerm16(std::vector<uint8_t> dom, std::vector<uint8_t> rng,
            size_t = 0 /* unused */) : PTransf16(dom, rng) {}
    PPerm16(std::initializer_list<uint8_t> il) : PTransf16(il) {}
    PPerm16 &operator=(const PPerm16 &) = default;

    static HPCOMBI_CONSTEXPR PPerm16 one() { return epu8id; }
    PPerm16 inline operator*(const PPerm16 &p) const {
        return this->PTransf16::operator*(p);
        return static_cast<PTransf16>(v) * static_cast<PTransf16>(p.v);
    }

    PPerm16 right_one() const { return PTransf16::right_one(); }
    PPerm16 left_one() const { return PTransf16::left_one(); }

    /** @copydoc common_inverse
     *  @par Algorithm:
     *  @f$O(n)@f$ algorithm using reference cast to arrays
     */
    inline PPerm16 inverse_ref() const;
    /** @copydoc common_inverse
     *  @par Algorithm:
     *  @f$O(\log n)@f$ algorithm using some kind of vectorized dichotomic
     * search.
     */
    inline PPerm16 inverse_find() const;
    /** @copydoc common_inverse
     *  @par Algorithm:
     *
     * Raise \e *this to power @f$\text{LCM}(1, 2, ..., n) - 1@f$ so complexity
     * is in @f$O(log (\text{LCM}(1, 2, ..., n) - 1)) = O(n)@f$
     */
};

/** Permutations of @f$\{0\dots 15\}@f$
 *
 */
struct Perm16 : public Transf16 /* public PPerm : diamond problem */ {

    Perm16() = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Perm16(const Perm16 &) = default;
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Perm16(const vect v) : Transf16(v) {}
    HPCOMBI_CONSTEXPR_CONSTRUCTOR Perm16(const epu8 x) : Transf16(x) {}
    Perm16 &operator=(const Perm16 &) = default;
    Perm16(std::initializer_list<uint8_t> il) : Transf16(il) {}
    explicit Perm16(uint64_t compressed) : Transf16(compressed) {}

    Perm16 inline operator*(const Perm16 &p) const {
        return HPCombi::permuted(v, p.v);
    }

    /** @class common_inverse
     * @brief The inverse permutation
     * @details
     * @returns the inverse of \c *this
     * @par Example:
     * @code
     * Perm16 x = {0,3,2,4,1,5,6,7,8,9,10,11,12,13,14,15};
     * x.inverse()
     * @endcode
     * Returns
     * @verbatim {0,4,2,1,3,5,6,7,8,9,10,11,12,13,14,15} @endverbatim
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
     *  @f$O(\log n)@f$ algorithm using some kind of vectorized dichotomic
     * search.
     */
    Perm16 inverse_find() const { return permutation_of(v, one()); }
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
    inline Perm16 inverse() const { return inverse_cycl(); }

    // It's not possible to have a static constexpr member of same type as class
    // being defined (see https://stackoverflow.com/questions/11928089/)
    // therefore we chose to have functions.
    static HPCOMBI_CONSTEXPR Perm16 one() { return epu8id; }

    inline static Perm16 elementary_transposition(uint64_t i);
    inline static Perm16 random();
    inline static Perm16 unrankSJT(int n, int r);

    /** @class common_lehmer
     * @brief The Lehmer code of a permutation
     * @details
     * @returns the Lehmer code of \c *this
     * @par Example:
     * @code
     * Perm16 x = {0,3,2,4,1,5,6,7,8,9,10,11,12,13,14,15};
     * x.lehmer()
     * @endcode
     * Returns
     * @verbatim {0,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0} @endverbatim
     */
    /** @copydoc common_lehmer
     *  @par Algorithm:
     *  Reference @f$O(n^2)@f$ algorithm using loop and indexed access
     */
    inline epu8 lehmer_ref() const;
    /** @copydoc common_lehmer
     *  @par Algorithm:
     *  Reference @f$O(n^2)@f$ algorithm using array, loop and indexed access
     */
    inline epu8 lehmer_arr() const;
    /** @copydoc common_lehmer
     *  @par Algorithm:
     *  Reference @f$O(n)@f$ algorithm using vector comparison
     */
    inline epu8 lehmer() const;
    inline uint8_t length_ref() const;
    inline uint8_t length_arr() const;
    inline uint8_t length() const;

    inline uint8_t nb_descents_ref() const;
    inline uint8_t nb_descents() const;

    inline uint8_t nb_cycles_ref() const;
    inline epu8 cycles_mask_unroll() const;
    inline uint8_t nb_cycles_unroll() const;
    inline uint8_t nb_cycles() const { return nb_cycles_unroll(); }

    /** @class common_left_weak_leq
     * @brief Compare two permutations for the left weak order
     * @par Example:
     * @code
     * Perm16 x{2,0,3,1}, y{3,0,2,1};
     * x.left_weak_leq(y)
     * @endcode
     * Returns @verbatim true @endverbatim
     */
    /** @copydoc common_left_weak_leq
     *  @par Algorithm:
     *  Reference @f$O(n^2)@f$ testing inclusion of inversions one by one
     */
    inline bool left_weak_leq_ref(Perm16 other) const;
    /** @copydoc common_left_weak_leq
     *  @par Algorithm:
     *  Reference @f$O(n)@f$ with vectorized test of inclusion
     */
    inline bool left_weak_leq(Perm16 other) const;
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

template <> struct hash<HPCombi::PPerm16> {
    inline size_t operator()(const HPCombi::PPerm16 &ar) const {
        return std::hash<HPCombi::epu8>{}(ar.v);
    }
};

template <> struct hash<HPCombi::Perm16> {
    inline size_t operator()(const HPCombi::Perm16 &ar) const {
        return uint64_t(ar);
    }
};

}  // namespace std

#endif  // HPCOMBI_PERM16_HPP_INCLUDED
