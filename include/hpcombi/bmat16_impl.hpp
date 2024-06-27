//****************************************************************************//
//    Copyright (C) 2018-2024 Finn Smith <fls3@st-andrews.ac.uk>              //
//    Copyright (C) 2018-2024 James Mitchell <jdm3@st-andrews.ac.uk>          //
//    Copyright (C) 2018-2024 Florent Hivert <Florent.Hivert@lisn.fr>,        //
//                                                                            //
//  This file is part of HP-Combi <https://github.com/libsemigroups/HPCombi>  //
//                                                                            //
//  HP-Combi is free software: you can redistribute it and/or modify it       //
//  under the terms of the GNU General Public License as published by the     //
//  Free Software Foundation, either version 3 of the License, or             //
//  (at your option) any later version.                                       //
//                                                                            //
//  HP-Combi is distributed in the hope that it will be useful, but WITHOUT   //
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or     //
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License      //
//  for  more details.                                                        //
//                                                                            //
//  You should have received a copy of the GNU General Public License along   //
//  with HP-Combi. If not, see <https://www.gnu.org/licenses/>.               //
//****************************************************************************//

// This file contains an implementation of fast boolean matrices up to
// dimension 16 x 16.

namespace HPCombi {
static_assert(std::is_trivial<BMat16>(), "BMat16 is not a trivial class!");

static constexpr xpu16 line{0x800, 0x901, 0xa02, 0xb03, 0xc04, 0xd05, 0xe06, 0xf07, 0x800, 0x901, 0xa02, 0xb03, 0xc04, 0xd05, 0xe06, 0xf07};
static constexpr xpu16 block{0x200, 0x604, 0xa08, 0xe0c, 0x301, 0x705, 0xb09, 0xf0d, 0x200, 0x604, 0xa08, 0xe0c, 0x301, 0x705, 0xb09, 0xf0d};

inline BMat16::BMat16(uint64_t n0, uint64_t n1, uint64_t n2, uint64_t n3) noexcept {
    xpu64 tmp{n0, n1, n2, n3};
    _data = simde_mm256_shuffle_epi8(tmp, line);
}

inline bool BMat16::operator==(BMat16 const &that) const noexcept {
    xpu64 tmp = _data ^ that._data;
    return ((tmp[0] == 0) and 
           (tmp[1] == 0) and
           (tmp[2] == 0) and 
           (tmp[3] == 0));
}

inline BMat16 BMat16::to_line() const {
    return BMat16(simde_mm256_shuffle_epi8(_data, line));
}

inline BMat16 BMat16::to_block() const {
    return BMat16(simde_mm256_shuffle_epi8(_data, block));
}

inline BMat16 BMat16::transpose_block() const noexcept {
    xpu64 x = simde_mm256_set_epi64x(_data[3], _data[1], _data[2], _data[0]);
    xpu64 y = (x ^ (x >> 7)) & (xpu64{0xAA00AA00AA00AA, 0xAA00AA00AA00AA, 0xAA00AA00AA00AA, 0xAA00AA00AA00AA});
    x = x ^ y ^ (y << 7);
    y = (x ^ (x >> 14)) & (xpu64{0xCCCC0000CCCC, 0xCCCC0000CCCC, 0xCCCC0000CCCC, 0xCCCC0000CCCC});
    x = x ^ y ^ (y << 14);
    y = (x ^ (x >> 28)) & (xpu64{0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0});
    x = x ^ y ^ (y << 28);
    return BMat16(x);
}

static constexpr xpu16 rot{0x302, 0x504, 0x706, 0x908, 0xb0a, 0xd0c, 0xf0e, 0x100, 0x302, 0x504, 0x706, 0x908, 0xb0a, 0xd0c, 0xf0e, 0x100};
static constexpr xpu16 alt{0x200, 0x604, 0xa08, 0xe0c, 0x301, 0x705, 0xb09, 0xf0d, 0x200, 0x604, 0xa08, 0xe0c, 0x301, 0x705, 0xb09, 0xf0d};

inline BMat16 BMat16::mult_transpose(BMat16 const &that) const noexcept {
    xpu16 x = _data;
    xpu16 y1 = that._data;
    xpu16 y2 = simde_mm256_set_epi64x(that._data[1], that._data[0], that._data[3], that._data[2]);
    xpu16 zero = simde_mm256_setzero_si256();
    xpu16 data = simde_mm256_setzero_si256();
    xpu16 diag1{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000};
    xpu16 diag2{0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    for (int i = 0; i < 8; ++i) {
        data |= ((x & y1) != zero) & diag1;
        data |= ((x & y2) != zero) & diag2;
        y1 = simde_mm256_shuffle_epi8(y1, rot);
        y2 = simde_mm256_shuffle_epi8(y2, rot);
        diag1 = simde_mm256_shuffle_epi8(diag1, rot);
        diag2 = simde_mm256_shuffle_epi8(diag2, rot);
    }
    return BMat16(data);
}

inline BMat16 BMat16::random() {
    static std::random_device _rd;
    static std::mt19937 _gen(_rd());
    static std::uniform_int_distribution<uint64_t> _dist(0, 0xffffffffffffffff);

    return BMat16(_dist(_gen), _dist(_gen), _dist(_gen), _dist(_gen));
}

}