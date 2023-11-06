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
using epu8x2 = std::array<epu8, 2>;

inline xpu8 &from_epu8x2(epu8x2 &p) { return reinterpret_cast<xpu8 &>(p); }
inline const xpu8 &from_epu8x2(const epu8x2 &p) {
    return reinterpret_cast<const xpu8 &>(p);
}
inline epu8x2 &to_epu8x2(xpu8 &p) { return reinterpret_cast<epu8x2 &>(p); }
inline const epu8x2 &to_epu8x2(const xpu8 &p) {
    return reinterpret_cast<const epu8x2 &>(p);
}
// xpu8 to_xpu8(epu8x2 p) { return reinterpret_cast<xpu8 &>(p); }
// epu8x2 from_xpu8(xpu8 p) { return reinterpret_cast<epu8x2 &>(p); }
// __m256d to_m256d(epu8x2 p) { return reinterpret_cast<__m256d &>(p); }
// epu8x2 from_m256d(__m256d p) { return reinterpret_cast<epu8x2 &>(p); }

#ifdef SIMDE_X86_AVX_NATIVE
static_assert(alignof(xpu8) == 32,
              "xpu8 type is not properly aligned by the compiler !");
#else
static_assert(alignof(xpu8) == 16,
              "xpu8 type is not properly aligned by the compiler !");
#endif

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

/** Permuting a #HPCombi::xpu8
 * @details reference version using array indexing
 */
inline xpu8 permuted_ref(xpu8 a, xpu8 b) noexcept;
/** Permuting a #HPCombi::xpu8
 * @details
 *     unsafe version giving undefined results if x2[i] >= 32 for some i
 * @par Algorithm: uses SSE4.1 shuffle instruction
 */
inline xpu8 permuted_unsafe(xpu8 x1, xpu8 x2) noexcept;
/** Permuting a #HPCombi::xpu8
 * @details
 *     unsafe version giving undefined results if x2[i] >= 32 for some i
 * @par Algorithm: uses AVX2 permute instruction
 */
inline xpu8 permuted_avx2_unsafe(xpu8 x1, xpu8 x2) noexcept;
/** Permuting a #HPCombi::xpu8
 * @details
  * Algorithm: call #HPCombi::permuted_unsafe after cleaning up the input
 */
inline xpu8 permuted(xpu8 x1, xpu8 x2) noexcept {
    return permuted_unsafe(x1, x2 & Xpu8(0x1f));
}

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
