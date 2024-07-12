//****************************************************************************//
//     Copyright (C) 2016-2024 Florent Hivert <Florent.Hivert@lisn.fr>,       //
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

#include <cstddef>   // for size_t
#include <cstdint>   // for uint64_t
#include <iostream>  // for char_traits, ostream, ostrin...
#include <string>    // for operator==
#include <utility>   // for pair
#include <vector>    // for vector, allocator

#include "test_main.hpp"                 // for TEST_AGREES, TEST_AGREES2
#include <catch2/catch_test_macros.hpp>  // for operator""_catch_sr, operator==

#include "hpcombi/bmat16.hpp"   // for BMat16, operator<<
#include "hpcombi/perm16.hpp"  // for Perm16
#include "hpcombi/vect16.hpp"  // for Vect16

namespace HPCombi {
namespace {
struct BMat16Fixture {
    const BMat16 zero, one1, one2, ones, bm, bm1, bmm1, bm2, bm2t, bm3, bm3t;
    const std::vector<BMat16> BMlist;
    BMat16Fixture()
        : zero(0, 0, 0, 0), one1(0, 0, 0, 1), one2(0, 0, 0, 0x20001), 
        ones(0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff),
          bm({{0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0},
              {0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1},
              {1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1},
              {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0},
              {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1},
              {1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0},
              {1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0},
              {1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0},
              {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
              {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1},
              {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1},
              {1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0},
              {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1},
              {1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0},
              {0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1},
              {0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0}}),
          bm1({{0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0},
               {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
               {0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0},
               {0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0},
               {1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1},
               {1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0},
               {1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0},
               {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1},
               {0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1},
               {1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0},
               {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1},
               {1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0},
               {1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1},
               {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0},
               {1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1},
               {0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}}),
          bmm1({{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1},
                {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1},
                {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1},
                {0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1},
                {0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1}}),
          bm2({{1, 1}, {0, 1}}), bm2t({{1, 0}, {1, 1}}),
          bm3({{0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1},
               {0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
               {1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0},
               {0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1},
               {1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0},
               {1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0},
               {0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1},
               {0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0},
               {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
               {1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0},
               {1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1},
               {1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
               {0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0},
               {0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1},
               {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0},
               {0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1}}),
          bm3t({{0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1},
                {0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0},
                {1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1},
                {0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
                {1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1},
                {0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1},
                {0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1},
                {0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1},
                {1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1},
                {0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1},
                {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                {1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0},
                {0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1},
                {1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
                {1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1}}),
          BMlist(
              {zero, one1, one2, ones, bm, bm1, bmm1, bm2, bm2t, bm3, bm3t}) {}
};
}  // namespace

//****************************************************************************//
//****************************************************************************//

TEST_CASE_METHOD(BMat16Fixture, "BMat16::transpose", "[BMat16][000]") {
    CHECK(zero.transpose() == zero);
    CHECK(bm2.transpose() == bm2t);
    CHECK(bm3.transpose() == bm3t);

    for (auto m : BMlist) {
        CHECK(m.transpose().transpose() == m);
    }
}

TEST_AGREES(BMat16Fixture, transpose, transpose_naive, BMlist, "[BMat16][001]");

TEST_CASE_METHOD(BMat16Fixture, "BMat16::operator*", "[BMat16][002]") {
    BMat16 tmp = bm * bm1;
    CHECK(tmp == bmm1);
    CHECK(tmp == bm * bm1);

    for (auto b : BMlist) {
        CHECK(zero * b == zero);
        CHECK(b * zero == zero);
        CHECK(b * b.one() == b);
        CHECK(b.one() * b == b);
        CHECK((b * b) * (b * b) == b * b * b * b);
    }

    for (auto b1 : BMlist) {
        for (auto b2 : BMlist) {
            for (auto b3 : BMlist) {
                CHECK((b1 * b2) * b3 == b1 * (b2 * b3));
            }
        }
    }
}

TEST_AGREES2(BMat16Fixture, BMat16::operator*, mult_naive, BMlist, "[BMat16][003]");
TEST_AGREES2(BMat16Fixture, BMat16::operator*, mult_naive_array, BMlist, "[BMat16][004]");

TEST_CASE("BMat16::random", "[BMat16][005]") {
    for (size_t d = 1; d < 8; ++d) {
        BMat16 bm = BMat16::random(d);
        for (size_t i = d + 1; i < 16; ++i) {
            for (size_t j = 0; j < 16; ++j) {
                CHECK(bm(i, j) == 0);
                CHECK(bm(j, i) == 0);
            }
        }
    }
}

TEST_CASE("BMat8::operator()", "[BMat8][006]") {
    std::vector<std::vector<bool>> mat = {
        {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1},
        {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1},
        {0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
        {0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1},
        {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1},
        {0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1},
        {0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1},
        {1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0}};
    BMat16 bm(mat);
    for (size_t i = 0; i < 15; ++i) {
        for (size_t j = 0; j < 15; ++j) {
            CHECK(static_cast<size_t>(bm(i, j)) == mat[i][j]);
        }
    }
}

TEST_CASE_METHOD(BMat16Fixture, "BMa16::operator<<", "[BMat16][007]") {
    std::ostringstream oss;
    oss << bm3;
    CHECK(oss.str() == "0001010001011011\n"
                       "0111011000110001\n"
                       "1010001101100010\n"
                       "0011010100001011\n"
                       "1001001111001110\n"
                       "1010100111101100\n"
                       "0110111001100001\n"
                       "0000010110111100\n"
                       "1100110100000001\n"
                       "1101000100001100\n"
                       "1101111000101101\n"
                       "1010000001000011\n"
                       "0011101110111000\n"
                       "0001001010011001\n"
                       "0100100100011110\n"
                       "0101011111110101\n");

    std::stringbuf buff;
    std::ostream os(&buff);
    os << BMat8::random();  // Also does not do anything visible
}

}  // namespace HPCombi
