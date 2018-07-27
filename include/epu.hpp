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

inline constexpr uint8_t operator "" _u8(unsigned long long arg) noexcept {
    return static_cast<uint8_t>(arg);
}

/// SIMD vector of 16 unsigned bytes
using epu8 = uint8_t __attribute__((vector_size(16)));
/// SIMD vector of 32 unsigned bytes
using xpu8 = uint8_t __attribute__((vector_size(32)));


/// Factory object for various SIMD constants in particular constexpr
template <class TPU> struct TPUBuild {

    using type_elem = typename std::remove_reference<decltype((TPU{})[0])>::type;
    static constexpr size_t size_elem = sizeof(type_elem);
    static constexpr size_t size = sizeof(TPU)/size_elem;
    using array = std::array<type_elem, size>;

    template <class Fun, std::size_t... Is> static HPCOMBI_CONSTEXPR
    TPU make_helper(Fun f, std::index_sequence<Is...>) { return {f(Is)...}; }

    // This is a handmade C++11 constexpr lambda;
    struct ConstFun {
        HPCOMBI_CONSTEXPR ConstFun(type_elem cc) : c(cc) {}
        HPCOMBI_CONSTEXPR type_elem operator()(type_elem) {return c;}
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

};



TPUBuild<epu8> Epu8;

// The following functions should be constexpr lambdas writen directly in
// their corresponding methods. However until C++17, constexpr lambda are
// forbidden. So we put them here.

/// The image of i by the identity function
HPCOMBI_CONSTEXPR uint8_t id_fun(uint8_t i) { return i; }
HPCOMBI_CONSTEXPR epu8 epu8id = Epu8(id_fun);
/// The image of i by the left cycle function
HPCOMBI_CONSTEXPR uint8_t left_cycle_fun(uint8_t i) { return (i + 15) % 16; }
HPCOMBI_CONSTEXPR epu8 left_cycle = Epu8(left_cycle_fun);
/// The image of i by the right cycle function
HPCOMBI_CONSTEXPR
uint8_t right_cycle_fun(uint8_t i) { return (i + 1) % 16; }
HPCOMBI_CONSTEXPR epu8 right_cycle = Epu8(right_cycle_fun);
/// The image of i by a left shift duplicating the hole
HPCOMBI_CONSTEXPR
uint8_t left_dup_fun(uint8_t i) { return i == 15 ? 15 : i + 1; }
HPCOMBI_CONSTEXPR epu8 left_dup = Epu8(left_dup_fun);
HPCOMBI_CONSTEXPR
uint8_t right_dup_fun(uint8_t i) { return i == 0 ? 0 : i - 1; }
HPCOMBI_CONSTEXPR epu8 right_dup = Epu8(right_dup_fun);


TPUBuild<epu8>::array &as_array(epu8 &v) {
    return reinterpret_cast<typename TPUBuild<epu8>::array &>(v);
}
const TPUBuild<epu8>::array &as_array(const epu8 &v) {
    return reinterpret_cast<const typename TPUBuild<epu8>::array &>(v);
}

inline uint64_t first_diff(epu8 a, epu8 b, size_t bound = 16);
inline uint64_t last_diff(epu8 a, epu8 b, size_t bound = 16);

inline bool equal(epu8 a, epu8 b);
inline bool not_equal(epu8 a, epu8 b);

inline bool less(epu8 a, epu8 b);
inline char less_partial(epu8 a, epu8 b, int k);

inline epu8 permuted(epu8 a, epu8 b);
inline epu8 shifted_right(epu8 a);
inline epu8 shifted_left(epu8 a);

inline bool is_sorted(epu8 a);
inline epu8 sorted(epu8 a);
inline epu8 sorted8(epu8 a);
inline epu8 revsorted(epu8 a);
inline epu8 revsorted8(epu8 a);

const uint64_t prime = 0x9e3779b97f4a7bb9;

}  // namespace HPCombi

namespace std {

inline std::ostream &operator<<(std::ostream &stream, HPCombi::epu8 const &a);
// We also specialize the templates equal_to, not_equal_to and hash

}  // namespace std


#include "epu_impl.hpp"

#endif  // HPCOMBI_EPU_HPP_INCLUDED
