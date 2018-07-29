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

#ifndef HPCOMBI_EPU_HPP_INCLUDED
#define HPCOMBI_EPU_HPP_INCLUDED

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>, equal_to<>
#include <ostream>
#include <iomanip>
#include <x86intrin.h>

#ifdef HPCOMBI_HAVE_CONFIG
#include "HPCombi-config.h"
#endif

#if __cplusplus <= 201103L
#include "seq.hpp"
#endif

#ifdef HPCOMBI_CONSTEXPR_FUN_ARGS
#define HPCOMBI_CONSTEXPR constexpr
#define HPCOMBI_CONSTEXPR_CONSTRUCTOR constexpr
#else
#pragma message "Using a constexpr broken compiler ! "                         \
                "Performance may not be optimal"
#define HPCOMBI_CONSTEXPR const
#define HPCOMBI_CONSTEXPR_CONSTRUCTOR
#endif

namespace HPCombi {

/// Unsigned 8 bits int constant.
inline constexpr uint8_t operator "" _u8(unsigned long long arg) noexcept {
    return static_cast<uint8_t>(arg);
}

/// SIMD vector of 16 unsigned bytes
using epu8 = uint8_t __attribute__((vector_size(16)));
/// SIMD vector of 32 unsigned bytes
using xpu8 = uint8_t __attribute__((vector_size(32)));



namespace {  // Implementation detail code

/// Factory object for various SIMD constants in particular constexpr
template <class TPU> struct TPUBuild {

    using type_elem = typename std::remove_reference<decltype((TPU{})[0])>::type;
    static constexpr size_t size_elem = sizeof(type_elem);
    static constexpr size_t size = sizeof(TPU)/size_elem;
    using array = std::array<type_elem, size>;

    template <class Fun, std::size_t... Is> static HPCOMBI_CONSTEXPR
    TPU make_helper(Fun f, std::index_sequence<Is...>) { return TPU{f(Is)...}; }

    /// A handmade C++11 constexpr lambda
    struct ConstFun {
        HPCOMBI_CONSTEXPR ConstFun(type_elem cc) : c(cc) {}
        HPCOMBI_CONSTEXPR type_elem operator()(type_elem) const {return c;}
        type_elem c;
    };

    inline TPU operator()(std::initializer_list<type_elem>, type_elem) const;

    template <class Fun>
    inline HPCOMBI_CONSTEXPR TPU operator()(Fun f) const {
        return make_helper(f, std::make_index_sequence<size>{});
    }

    inline HPCOMBI_CONSTEXPR TPU operator()(type_elem c) const {
        return make_helper(ConstFun(c), std::make_index_sequence<size>{});
    }
    // explicit overloading for int constants
    inline HPCOMBI_CONSTEXPR TPU operator()(int c) const {
        return operator()(uint8_t(c));
    }

};

// The following functions should be constexpr lambdas writen directly in
// their corresponding methods. However until C++17, constexpr lambda are
// forbidden. So we put them here.
/// The image of i by the identity function
HPCOMBI_CONSTEXPR uint8_t id_fun(uint8_t i) { return i; }
/// The image of i by the left cycle function
HPCOMBI_CONSTEXPR uint8_t left_cycle_fun(uint8_t i) { return (i + 15) % 16; }
/// The image of i by the right cycle function
HPCOMBI_CONSTEXPR
uint8_t right_cycle_fun(uint8_t i) { return (i + 1) % 16; }
/// The image of i by a left shift duplicating the hole
HPCOMBI_CONSTEXPR
uint8_t left_dup_fun(uint8_t i) { return i == 15 ? 15 : i + 1; }
/// The image of i by a right shift duplicating the hole
HPCOMBI_CONSTEXPR
uint8_t right_dup_fun(uint8_t i) { return i == 0 ? 0 : i - 1; }
/// The complement of i to 15
HPCOMBI_CONSTEXPR
uint8_t complement_fun(uint8_t i) { return 15 - i; }

}  // Anonymous namespace



/// Factory object for various SIMD constants in particular constexpr
TPUBuild<epu8> Epu8;

/// The indentity #HPCombi::epu8
HPCOMBI_CONSTEXPR epu8 epu8id = Epu8(id_fun);
/// The reverted identity #HPCombi::epu8
HPCOMBI_CONSTEXPR epu8 epu8rev = Epu8(complement_fun);
/// Left cycle #HPCombi::epu8 permutation
HPCOMBI_CONSTEXPR epu8 left_cycle = Epu8(left_cycle_fun);
/// Right cycle #HPCombi::epu8 permutation
HPCOMBI_CONSTEXPR epu8 right_cycle = Epu8(right_cycle_fun);
/// Left shift #HPCombi::epu8, duplicating the rightmost entry
HPCOMBI_CONSTEXPR epu8 left_dup = Epu8(left_dup_fun);
/// Right shift #HPCombi::epu8, duplicating the leftmost entry
HPCOMBI_CONSTEXPR epu8 right_dup = Epu8(right_dup_fun);


/** Cast a #HPCombi::epu8 to a c++ \c std::array
 *
 *  This is usually faster for algorithm using a lot of indexed acces.
 */
inline TPUBuild<epu8>::array &as_array(epu8 &v) {
    return reinterpret_cast<typename TPUBuild<epu8>::array &>(v);
}
/** Cast a constant #HPCombi::epu8 to a C++ \c std::array
 *
 *  This is usually faster for algorithm using a lot of indexed acces.
 */
inline const TPUBuild<epu8>::array &as_array(const epu8 &v) {
    return reinterpret_cast<const typename TPUBuild<epu8>::array &>(v);
}
/** Cast a C++ \c std::array to a #HPCombi::epu8 */
// Passing the argument by reference triggers a segfault in gcc
// Since vector types doesn't belongs to the standard, I didn't manage
// to know if I'm using undefined behavior here.
inline epu8 from_array(TPUBuild<epu8>::array a) {
    return reinterpret_cast<const epu8 &>(a);
}


/** Test whether all the entries of a #HPCombi::epu8 are zero */
inline bool is_all_zero(epu8 a) { return _mm_testz_si128(a, a); }
/** Equality of #HPCombi::epu8 */
inline bool equal(epu8 a, epu8 b) { return is_all_zero(_mm_xor_si128(a, b)); }
/** Non equality of #HPCombi::epu8 */
inline bool not_equal(epu8 a, epu8 b) { return not equal(a, b); }

/** Permuting a #HPCombi::epu8 */
inline epu8 permuted(epu8 a, epu8 b) { return _mm_shuffle_epi8(a, b); }
/** Left shifted of a #HPCombi::epu8 inserting a 0
 * @warning we use the convention that the 0 entry is on the left !
 */
inline epu8 shifted_right(epu8 a) { return _mm_bslli_si128(a,1); }
/** Right shifted of a #HPCombi::epu8 inserting a 0
 * @warning we use the convention that the 0 entry is on the left !
 */
inline epu8 shifted_left(epu8 a) { return _mm_bsrli_si128(a,1); }
/** Reverting a #HPCombi::epu8 */
inline epu8 reverted(epu8 a) { return permuted(a, epu8rev); }

/** The first difference between two #HPCombi::epu8 */
inline uint64_t first_diff(epu8 a, epu8 b, size_t bound = 16);
/** The last difference between two #HPCombi::epu8 */
inline uint64_t last_diff(epu8 a, epu8 b, size_t bound = 16);
/** Lexicographic comparison between two #HPCombi::epu8 */
inline bool less(epu8 a, epu8 b);
/** Partial lexicographic comparison between two #HPCombi::epu8
 * @param a, b : the vectors to compare
 * @param k : the bound for the lexicographic comparison
 * @return a positive, negative or zero char depending on the result
 */
inline char less_partial(epu8 a, epu8 b, int k);

/** Vector min between two #HPCombi::epu8 0 */
inline epu8 min(epu8 a, epu8 b) { return _mm_min_epu8(a, b); }
/** Vector max between two #HPCombi::epu8 0 */
inline epu8 max(epu8 a, epu8 b) { return _mm_max_epu8(a, b); }

/** Testing if a #HPCombi::epu8 is sorted */
inline bool is_sorted(epu8 a);
/** Return a sorted #HPCombi::epu8
 * @details
 * @par Algorithm:
 * Uses the 9 stages sorting network #sorting_rounds
 */
inline epu8 sorted(epu8 a);
/** Return a #HPCombi::epu8 with the two half sorted
 * @details
 * @par Algorithm: Uses a 6stages sorting network #sorting_rounds8
 */
inline epu8 sorted8(epu8 a);
/** Return a reverse sorted #HPCombi::epu8
 * @details
 * @par Algorithm:
 * Uses the 9 stages sorting network #sorting_rounds
 */
inline epu8 revsorted(epu8 a);
/** Return a #HPCombi::epu8 with the two half reverse sorted
 * @details
 * @par Algorithm: Uses a 6stages sorting network #sorting_rounds8
 */
inline epu8 revsorted8(epu8 a);

/** A prime number good for hashing */
const uint64_t prime = 0x9e3779b97f4a7bb9;

/** A random #HPCombi::epu8
 * @details
 * @param bnd : the upper bound for the value of the entries
 * @returns a random #HPCombi::epu8 with value in the interval @f$[0, 1, 2, ..., bnd-1]@f$.
 */
inline epu8 random_epu8(uint16_t bnd);

/** Remove duplicates in a sorted #HPCombi::epu8
 * @details
 * @param a: supposed to be sorted
 * @param repl: the value replacing the duplicate entries (default to 0)
 * @return a where repeated occurences of entries are replaced by \c repl
 */
inline epu8 remove_dups(epu8 a, uint8_t repl=0);


inline uint8_t horiz_sum_ref(epu8);
inline uint8_t horiz_sum4(epu8);
inline uint8_t horiz_sum3(epu8);
inline uint8_t horiz_sum(epu8 v) { return horiz_sum3(v); }

inline epu8 partial_sums_ref(epu8);
inline epu8 partial_sums_round(epu8);
inline epu8 partial_sums(epu8 v) { return partial_sums_round(v); }

}  // namespace HPCombi

namespace std {

inline std::ostream &operator<<(std::ostream &stream, HPCombi::epu8 const &a);

// We also specialize the templates equal_to, not_equal_to and hash

using HPCombi::epu8;
/** Vector min between two #HPCombi::epu8 0 */
inline epu8 min(epu8 a, epu8 b) { return _mm_min_epu8(a, b); }
/** Vector max between two #HPCombi::epu8 0 */
inline epu8 max(epu8 a, epu8 b) { return _mm_max_epu8(a, b); }

}  // namespace std


#include "epu_impl.hpp"

#endif  // HPCOMBI_EPU_HPP_INCLUDED
