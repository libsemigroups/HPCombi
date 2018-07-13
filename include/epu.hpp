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

inline constexpr uint8_t operator "" _u8( unsigned long long arg ) noexcept {
    return static_cast<uint8_t >( arg );
}

/// SIMD vector of 16 unsigned bytes
using epu8 = uint8_t __attribute__((vector_size(16)));
/// SIMD vector of 32 unsigned bytes
using xpu8 = uint8_t __attribute__((vector_size(32)));

/// Factory object for constructing various SIMD constants in particular
/// constexpr
template <class TPU> struct TPUBuild {

    using type_elem = typename std::remove_reference<decltype((TPU{})[0])>::type;
    static constexpr size_t size_elem = sizeof(type_elem);
    static constexpr size_t size = sizeof(TPU)/size_elem;

    template <class Function, std::size_t... Indices>
    static HPCOMBI_CONSTEXPR
    TPU make_helper(Function f, std::index_sequence<Indices...>) {
        return TPU{f(Indices)...};
    }

    template <type_elem c> static HPCOMBI_CONSTEXPR
    type_elem constfun(type_elem) { return c; }

    template <type_elem c> static HPCOMBI_CONSTEXPR TPU cst() {
        return make_helper(constfun<c>, std::make_index_sequence<size>{});
    }

    static std::array<type_elem, size> &as_array(TPU &v) {
        return reinterpret_cast<std::array<type_elem, size> &>(v);
    }

    static const std::array<type_elem, size> &as_array(const TPU &v) {
        return reinterpret_cast<const std::array<type_elem, size> &>(v);
    }

    inline TPU operator()(std::initializer_list<type_elem> il,
                            type_elem def) {
        TPU res;
        assert(il.size() <= size);
        std::copy(il.begin(), il.end(), as_array(res).begin());
        for (size_t i = il.size(); i < size; ++i)
            res[i] = def;
        return res;
    }

    template <class Function>
    inline HPCOMBI_CONSTEXPR TPU operator()(Function f) const {
        return make_helper(f, std::make_index_sequence<size>{});
    }
};

// The following functions should be constexpr lambdas writen directly in
// their corresponding methods. However until C++17, constexpr lambda are
// forbidden. So we put them here.

/// The image of i by the identity function
HPCOMBI_CONSTEXPR
uint8_t id_fun(uint8_t i) { return i; }
/// The image of i by the left cycle function
HPCOMBI_CONSTEXPR
uint8_t left_cycle_fun(uint8_t i) { return (i + 15) % 16; }
/// The image of i by the right cycle function
HPCOMBI_CONSTEXPR
uint8_t right_cycle_fun(uint8_t i) { return (i + 1) % 16; }
/// The image of i by a left shift filling the hole with a @p 0xff
HPCOMBI_CONSTEXPR
uint8_t left_shift_ff_fun(uint8_t i) { return i == 15 ? 0xff : i + 1; }
HPCOMBI_CONSTEXPR
uint8_t right_shift_ff_fun(uint8_t i) { return i == 0 ? 0xff : i - 1; }
/// The image of i by a left shift duplicating the hole
HPCOMBI_CONSTEXPR
uint8_t left_shift_fun(uint8_t i) { return i == 15 ? 15 : i + 1; }
HPCOMBI_CONSTEXPR
uint8_t right_shift_fun(uint8_t i) { return i == 0 ? 0 : i - 1; }

TPUBuild<epu8> epu8cons;

HPCOMBI_CONSTEXPR epu8 epu8id = epu8cons(id_fun);
HPCOMBI_CONSTEXPR epu8 right_shift = epu8cons(right_shift_fun);
HPCOMBI_CONSTEXPR epu8 left_shift = epu8cons(left_shift_fun);

// Old Clang doesn't automatically broadcast uint8_t into epu8
// We therefore write there the explicit constants
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x00 = epu8cons.cst<0x00_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x01 = epu8cons.cst<0x01_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x02 = epu8cons.cst<0x02_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x04 = epu8cons.cst<0x04_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x08 = epu8cons.cst<0x08_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x0F = epu8cons.cst<0x0F_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0x10 = epu8cons.cst<0x10_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0xF0 = epu8cons.cst<0xF0_u8>();
HPCOMBI_CONSTEXPR epu8 cst_epu8_0xFF = epu8cons.cst<0xFF_u8>();


inline uint64_t first_diff(epu8 a, epu8 b, size_t bound = 16);
inline uint64_t last_diff(epu8 a, epu8 b, size_t bound = 16);

inline bool equal(epu8 a, epu8 b);
inline bool not_equal(epu8 a, epu8 b);

inline bool less(epu8 a, epu8 b);
inline char less_partial(epu8 a, epu8 b, int k);

inline epu8 permuted(epu8 a, epu8 b);

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
