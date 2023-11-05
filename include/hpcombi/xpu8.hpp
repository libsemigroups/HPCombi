////////////////////////////////////////////////////////////////////////////////
//     Copyright (C) 2023 Florent Hivert <Florent.Hivert@lri.fr>,             //
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
////////////////////////////////////////////////////////////////////////////////

#ifndef HPCOMBI_XPU8_HPP_INCLUDED
#define HPCOMBI_XPU8_HPP_INCLUDED

#include <array>             // for array
#include <cstddef>           // for size_t
#include <cstdint>           // for uint8_t, uint64_t, int8_t
#include <initializer_list>  // for initializer_list
#include <ostream>           // for ostream
#include <string>            // for string
#include <type_traits>       // for remove_reference_t
#include <utility>           // for make_index_sequence, ind...

#include "debug.hpp"         // for HPCOMBI_ASSERT
#include "builder.hpp"       // for TPUBuild
#include "vect_generic.hpp"  // for VectGeneric
#include "epu8.hpp"

#include "simde/x86/avx.h"
#include "simde/x86/avx2.h"

namespace HPCombi {

/// SIMD vector of 32 unsigned bytes
using xpu8 = uint8_t __attribute__((vector_size(32)));

static_assert(alignof(xpu8) == 32,
              "xpu8 type is not properly aligned by the compiler !");

/** Factory object acting as a class constructor for type #HPCombi::xpu8.
 * see #HPCombi::TPUBuild for usage and capability
 */
constexpr TPUBuild<xpu8> Xpu8 {};

/** Test whether all the entries of a #HPCombi::xpu8 are zero */
inline bool is_all_zero(xpu8 a) noexcept {
    return simde_mm256_testz_si256(a, a); }
/** Test whether all the entries of a #HPCombi::xpu8 are one */
inline bool is_all_one(xpu8 a) noexcept {
    return simde_mm256_testc_si256(a, Xpu8(0xFF));
}

/** Equality of #HPCombi::xpu8 */
inline bool equal(xpu8 a, xpu8 b) noexcept {
    return is_all_zero(simde_mm256_xor_si256(a, b));
}
/** Non equality of #HPCombi::xpu8 */
inline bool not_equal(xpu8 a, xpu8 b) noexcept {
    return !equal(a, b);
}

/** Vector min between two #HPCombi::xpu8 0 */
inline xpu8 min(xpu8 a, xpu8 b) noexcept { return simde_mm256_min_epu8(a, b); }
/** Vector max between two #HPCombi::xpu8 0 */
inline xpu8 max(xpu8 a, xpu8 b) noexcept { return simde_mm256_max_epu8(a, b); }


}  // namespace HPCombi

namespace std {

inline std::ostream &operator<<(std::ostream &stream, HPCombi::xpu8 const &a);

inline std::string to_string(HPCombi::xpu8 const &a);

/** We also specialize the struct
 *  - std::equal_to<xpu8>
 *  - std::not_equal_to<xpu8>
 */

}  // namespace HPCombi

#include "xpu8_impl.hpp"

#endif  // HPCOMBI_XPU8_HPP_INCLUDED
