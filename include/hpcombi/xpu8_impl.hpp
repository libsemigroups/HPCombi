////////////////////////////////////////////////////////////////////////////////
//       Copyright (C) 2023 Florent Hivert <Florent.Hivert@lri.fr>,           //
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

// This is the implementation part of xpu8.hpp this should be seen as
// implementation details and should not be included directly.

#include <initializer_list>
#include <iostream>
#include <random>
#include <sstream>

#include "vect_generic.hpp"


namespace HPCombi {

inline xpu8 permuted(xpu8 x1, xpu8 x2) noexcept {
    const epu8x2 &v1 = to_epu8x2(x1);
    x2 = x2 & Xpu8(0x1f);
    // std::cout << x2 << std::endl;
    const epu8x2 &v2 = to_epu8x2(x2);
    return from_epu8x2({
            _mm_blendv_epi8(_mm_shuffle_epi8(v1[1], v2[0]),
                            _mm_shuffle_epi8(v1[0], v2[0]), v2[0] < 16),
            _mm_blendv_epi8(_mm_shuffle_epi8(v1[1], v2[1]),
                            _mm_shuffle_epi8(v1[0], v2[1]), v2[1] < 16)});
}

inline xpu8 permuted_ref(xpu8 a, xpu8 b) noexcept {
    xpu8 res;
    for (uint64_t i = 0; i < 32; i++)
        res[i] = a[b[i] & 0x1f];
    return res;
}

}  // namespace HPCombi

namespace std {

inline std::ostream &operator<<(std::ostream &stream, const HPCombi::xpu8 &a) {
    return HPCombi::ostream_insert(stream, a);
}

inline std::string to_string(const HPCombi::xpu8 &a) {
    std::ostringstream ss;
    ss << a;
    return ss.str();
}

template <> struct equal_to<HPCombi::xpu8> {
    bool operator()(const HPCombi::xpu8 &lhs,
                    const HPCombi::xpu8 &rhs) const noexcept {
        return HPCombi::equal(lhs, rhs);
    }
};

template <> struct not_equal_to<HPCombi::xpu8> {
    bool operator()(const HPCombi::xpu8 &lhs,
                    const HPCombi::xpu8 &rhs) const noexcept {
        return HPCombi::not_equal(lhs, rhs);
    }
};

// template <> struct hash<HPCombi::xpu8> {
//     inline size_t operator()(HPCombi::xpu8 a) const noexcept {
//         unsigned __int128 v0 = simde_mm_extract_epi64(a, 0);
//         unsigned __int128 v1 = simde_mm_extract_epi64(a, 1);
//         return ((v1 * HPCombi::prime + v0) * HPCombi::prime) >> 64;

//         /* The following is extremely slow on Renner benchmark
//            uint64_t v0 = simde_mm_extract_epi64(ar.v, 0);
//            uint64_t v1 = simde_mm_extract_epi64(ar.v, 1);
//            size_t seed = v0 + 0x9e3779b9;
//            seed ^= v1 + 0x9e3779b9 + (seed<<6) + (seed>>2);
//            return seed;
//         */
//     }
// };

// template <> struct less<HPCombi::xpu8> {
//     // WARNING: due to endianness this is not lexicographic comparison,
//     //          but we don't care when using in std::set.
//     // 10% faster than calling the lexicographic comparison operator !
//     inline size_t operator()(const HPCombi::xpu8 &v1,
//                              const HPCombi::xpu8 &v2) const noexcept {
//         simde__m128 v1v = simde__m128(v1), v2v = simde__m128(v2);
//         return v1v[0] == v2v[0] ? v1v[1] < v2v[1] : v1v[0] < v2v[0];
//     }
// };

}  // namespace std
