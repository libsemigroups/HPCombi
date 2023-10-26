/******************************************************************************/
/*     Copyright (C) 2016-2018 Florent Hivert <Florent.Hivert@lri.fr>,        */
/*                                                                            */
/*  Distributed under the terms of the GNU General Public License (GPL)       */
/*                                                                            */
/*    This code is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       */
/*   General Public License for more details.                                 */
/*                                                                            */
/*  The full text of the GPL is available at:                                 */
/*                                                                            */
/*                  http://www.gnu.org/licenses/                              */
/******************************************************************************/

#include <iostream>
#include <sstream>
#include <vector>

#include "test_main.hpp"
#include <catch2/catch_test_macros.hpp>

#include "bmat8.hpp"
#include "epu.hpp"

namespace HPCombi {
namespace {
struct BMat8Fixture {
    const BMat8 zero, one1, one2, ones, bm, bm1, bmm1, bm2, bm2t, bm3, bm3t;
    const std::vector<BMat8> BMlist;
    BMat8Fixture()
        : zero(0), one1(1), one2(0x201), ones(0xffffffffffffffff),
          bm({{0, 0, 0, 1, 0, 0, 1, 1},
              {1, 1, 1, 1, 1, 1, 0, 1},
              {0, 1, 1, 1, 0, 1, 0, 1},
              {1, 1, 0, 1, 1, 1, 1, 1},
              {0, 0, 1, 0, 0, 1, 1, 1},
              {1, 1, 0, 0, 0, 0, 0, 1},
              {0, 1, 0, 0, 0, 0, 1, 1},
              {0, 1, 1, 1, 1, 0, 1, 0}}),
          bm1({{0, 0, 0, 1, 0, 0, 1, 1},
               {0, 0, 1, 0, 0, 1, 0, 1},
               {1, 1, 0, 0, 1, 1, 0, 1},
               {1, 1, 0, 0, 0, 0, 0, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {0, 1, 0, 1, 1, 1, 1, 1},
               {0, 1, 0, 1, 0, 1, 0, 1},
               {0, 1, 0, 0, 0, 0, 1, 0}}),
          bmm1({{1, 1, 0, 1, 0, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 0, 1, 1, 1, 1, 1},
                {0, 1, 1, 1, 0, 1, 1, 1},
                {0, 1, 1, 1, 0, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1}}),
          bm2({{1, 1}, {0, 1}}), bm2t({{1, 0}, {1, 1}}),
          bm3({{0, 0, 0, 1, 0, 0, 1, 1},
               {1, 1, 1, 1, 1, 1, 0, 1},
               {0, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 0, 1, 1, 1, 1, 1},
               {0, 0, 1, 0, 0, 1, 1, 1},
               {1, 1, 0, 0, 0, 0, 0, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {0, 1, 1, 1, 1, 0, 1, 0}}),
          bm3t({{0, 1, 0, 1, 0, 1, 0, 0},
                {0, 1, 1, 1, 0, 1, 1, 1},
                {0, 1, 1, 0, 1, 0, 0, 1},
                {1, 1, 1, 1, 0, 0, 0, 1},
                {0, 1, 1, 1, 0, 0, 0, 1},
                {0, 1, 1, 1, 1, 0, 0, 0},
                {1, 0, 0, 1, 1, 0, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 0}}),
          BMlist(
              {zero, one1, one2, ones, bm, bm1, bmm1, bm2, bm2t, bm3, bm3t}) {}
};
}  // namespace

//****************************************************************************//
//****************************************************************************//

TEST_CASE_METHOD(BMat8Fixture, "BMat8::one", "[BMat8][000]") {
    REQUIRE(BMat8::one(0) == zero);
    REQUIRE(BMat8::one(2) == BMat8({{1, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 1, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0}}));
    REQUIRE(BMat8::one(5) == BMat8({{1, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 1, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 1, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 1, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 1, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 0, 0, 0}}));
    REQUIRE(BMat8::one(8) == BMat8::one());
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::transpose", "[BMat8][001]") {

    REQUIRE(zero.transpose() == zero);
    REQUIRE(bm2.transpose() == bm2t);
    REQUIRE(bm3.transpose() == bm3t);

    for (auto m : BMlist) {
        REQUIRE(m.transpose().transpose() == m);
    }
}

TEST_AGREES(BMat8Fixture, BMat8, transpose, transpose_mask, BMlist,
            "[BMat8][002]");

TEST_AGREES(BMat8Fixture, BMat8, transpose, transpose_maskd, BMlist,
            "[BMat8][003]");

TEST_CASE_METHOD(BMat8Fixture, "BMat8::transpose2", "[BMat8][004]") {
    for (auto a : BMlist) {
        for (auto b : BMlist) {
            BMat8 at = a, bt = b;
            BMat8::transpose2(at, bt);
            REQUIRE(at == a.transpose());
            REQUIRE(bt == b.transpose());
        }
    }
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::operator*", "[BMat8][005]") {
    BMat8 tmp = bm * bm1;
    REQUIRE(tmp == bmm1);
    REQUIRE(tmp == bm * bm1);

    for (auto b : BMlist) {
        REQUIRE(zero * b == zero);
        REQUIRE(b * zero == zero);
        REQUIRE(b * b.one() == b);
        REQUIRE(b.one() * b == b);
        REQUIRE((b * b) * (b * b) == b * b * b * b);
    }

    for (auto b1 : BMlist) {
        for (auto b2 : BMlist) {
            for (auto b3 : BMlist) {
                REQUIRE((b1 * b2) * b3 == b1 * (b2 * b3));
            }
        }
    }
}

TEST_CASE("BMat8::random", "[BMat8][006]") {
    for (size_t d = 1; d < 8; ++d) {
        BMat8 bm = BMat8::random(d);
        for (size_t i = d + 1; i < 8; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                REQUIRE(bm(i, j) == 0);
                REQUIRE(bm(j, i) == 0);
            }
        }
    }
}

TEST_CASE("BMat8::operator()", "[BMat8][007]") {
    std::vector<std::vector<bool>> mat = {
        {0, 0, 0, 1, 0, 0, 1}, {0, 1, 1, 1, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1},
        {0, 0, 1, 0, 0, 1, 1}, {1, 1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1, 0, 1}};
    BMat8 bm(mat);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 7; ++j) {
            REQUIRE(static_cast<size_t>(bm(i, j)) == mat[i][j]);
        }
    }
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::operator<<", "[BMat8][008]") {
    std::ostringstream oss;
    oss << bm3;
    REQUIRE(oss.str() == "00010011\n"
                         "11111101\n"
                         "01111101\n"
                         "11011111\n"
                         "00100111\n"
                         "11000001\n"
                         "01000011\n"
                         "01111010\n");

    std::stringbuf buff;
    std::ostream os(&buff);
    os << BMat8::random();  // Also does not do anything visible
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::set", "[BMat8][009]") {
    BMat8 bs;
    bs = bm;
    bs.set(0, 0, 1);
    REQUIRE(bs != bm);
    bs = bm;
    bs.set(0, 0, 0);
    REQUIRE(bs == bm);
    bs = bm;
    bs.set(2, 4, 1);
    REQUIRE(bs != bm);
    REQUIRE(bs == bm3);

    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            bs.set(i, j, true);
    REQUIRE(bs == ones);

    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            bs.set(i, j, false);
    REQUIRE(bs == zero);
}

TEST_CASE("BMat8::row_space_basis", "[BMat8][010]") {
    BMat8 bm({{0, 1, 1, 1, 0, 1, 0, 1},
              {0, 0, 0, 0, 0, 0, 0, 1},
              {1, 1, 1, 1, 1, 1, 0, 1},
              {1, 1, 0, 1, 1, 1, 1, 1},
              {0, 0, 1, 0, 0, 1, 1, 1},
              {1, 1, 0, 0, 0, 0, 0, 1},
              {0, 1, 0, 0, 0, 0, 1, 1},
              {0, 1, 1, 1, 1, 0, 1, 0}});

    BMat8 bm2({{1, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 0, 1, 1, 1, 1, 1},
               {1, 1, 0, 0, 0, 0, 0, 1},
               {0, 1, 1, 1, 1, 0, 1, 0},
               {0, 1, 1, 1, 0, 1, 0, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {0, 0, 1, 0, 0, 1, 1, 1},
               {0, 0, 0, 0, 0, 0, 0, 1}});

    REQUIRE(bm.row_space_basis() == bm2.row_space_basis());

    BMat8 bm3({{1, 1, 1, 1, 0, 1, 0, 1},
               {0, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 0, 0, 1, 0, 1},
               {1, 1, 0, 0, 0, 1, 1, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {1, 0, 0, 0, 0, 1, 0, 0}});

    BMat8 bm4({{1, 1, 1, 1, 0, 1, 0, 1},
               {1, 1, 1, 0, 0, 1, 0, 1},
               {1, 0, 0, 0, 0, 1, 0, 0},
               {0, 1, 1, 1, 1, 1, 0, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0}});

    REQUIRE(bm3.row_space_basis() == bm4);
    REQUIRE(bm4.row_space_basis() == bm4);

    BMat8 bm5(0xff00000000000000);

    uint64_t data = 0xffffffffffffffff;

    for (size_t i = 0; i < 7; ++i) {
        REQUIRE(BMat8(data).row_space_basis() == bm5);
        data = data >> 8;
    }

    for (size_t i = 0; i < 1000; ++i) {
        bm = BMat8::random();
        REQUIRE(bm.row_space_basis().row_space_basis() == bm.row_space_basis());
    }
}

TEST_CASE("BMat8::col_space_basis", "[BMat8][011]") {
    BMat8 bm({{0, 1, 1, 1, 0, 1, 0, 1},
              {0, 0, 0, 0, 0, 0, 0, 1},
              {1, 1, 1, 1, 1, 1, 0, 1},
              {1, 1, 0, 1, 1, 1, 1, 1},
              {0, 0, 1, 0, 0, 1, 1, 1},
              {1, 1, 0, 0, 0, 0, 0, 1},
              {0, 1, 0, 0, 0, 0, 1, 1},
              {0, 1, 1, 1, 1, 0, 1, 0}});

    BMat8 bm2({{1, 1, 1, 1, 1, 0, 0, 0},
               {1, 0, 0, 0, 0, 0, 0, 0},
               {1, 1, 1, 1, 1, 1, 1, 0},
               {1, 1, 1, 1, 0, 1, 1, 1},
               {1, 1, 0, 0, 1, 0, 0, 1},
               {1, 0, 1, 0, 0, 1, 0, 0},
               {1, 0, 1, 0, 0, 0, 0, 1},
               {0, 0, 1, 1, 1, 0, 1, 1}});

    REQUIRE(bm.col_space_basis() == bm2);

    BMat8 bm3({{1, 1, 1, 1, 0, 1, 0, 1},
               {0, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 1, 1, 1, 0, 1},
               {1, 1, 1, 0, 0, 1, 0, 1},
               {1, 1, 0, 0, 0, 1, 1, 1},
               {0, 1, 0, 0, 0, 0, 1, 1},
               {1, 0, 0, 0, 0, 1, 0, 0}});

    BMat8 bm4({{1, 1, 1, 0, 0, 0, 0, 0},
               {1, 1, 0, 1, 0, 0, 0, 0},
               {1, 1, 1, 1, 0, 0, 0, 0},
               {1, 1, 1, 1, 0, 0, 0, 0},
               {1, 0, 1, 0, 0, 0, 0, 0},
               {0, 0, 1, 0, 1, 0, 0, 0},
               {0, 0, 0, 0, 1, 0, 0, 0},
               {0, 0, 1, 0, 0, 0, 0, 0}});

    REQUIRE(bm3.col_space_basis() == bm4);

    uint64_t col = 0x8080808080808080;
    BMat8 bm5(col);

    uint64_t data = 0xffffffffffffffff;

    for (size_t i = 0; i < 7; ++i) {
        REQUIRE(BMat8(data).col_space_basis() == bm5);
        data &= ~(col >> i);
    }

    for (size_t i = 0; i < 1000; ++i) {
        bm = BMat8::random();
        REQUIRE(bm.col_space_basis().col_space_basis() == bm.col_space_basis());
    }
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::row_space_size", "[BMat8][012]") {
    REQUIRE(zero.row_space_size() == 1);
    REQUIRE(one1.row_space_size() == 2);
    REQUIRE(one2.row_space_size() == 4);
    REQUIRE(BMat8::one().row_space_size() == 256);
    REQUIRE(bm.row_space_size() == 22);
    REQUIRE(bm1.row_space_size() == 31);
    REQUIRE(bm2.row_space_size() == 3);
    REQUIRE(bm2t.row_space_size() == 3);
    REQUIRE(bm3.row_space_size() == 21);
    REQUIRE(bm3t.row_space_size() == 21);
    REQUIRE(bmm1.row_space_size() == 6);
}

TEST_AGREES(BMat8Fixture, BMat8, row_space_size_ref, row_space_size, BMlist,
            "[BMat8][013]");
TEST_AGREES(BMat8Fixture, BMat8, row_space_size_ref, row_space_size_incl,
            BMlist, "[BMat8][014]");
TEST_AGREES(BMat8Fixture, BMat8, row_space_size_ref, row_space_size_incl1,
            BMlist, "[BMat8][015]");
TEST_AGREES(BMat8Fixture, BMat8, row_space_size_ref, row_space_size_bitset,
            BMlist, "[BMat8][016]");

TEST_CASE_METHOD(BMat8Fixture, "BMat8::row_space_included", "[BMat8][017]") {
    REQUIRE(zero.row_space_included(one1));
    REQUIRE_FALSE(one1.row_space_included(zero));

    BMat8 m1({{1, 1, 0}, {1, 0, 1}, {0, 0, 0}});
    BMat8 m2({{0, 0, 0}, {1, 0, 1}, {1, 1, 0}});
    REQUIRE(m1.row_space_included(m2));
    REQUIRE(m2.row_space_included(m1));

    BMat8 m3({{0, 0, 1}, {1, 0, 1}, {1, 1, 0}});
    REQUIRE(m1.row_space_included(m3));
    REQUIRE(m2.row_space_included(m3));
    REQUIRE_FALSE(m3.row_space_included(m1));
    REQUIRE_FALSE(m3.row_space_included(m1));

    REQUIRE(m1.row_space_included(BMat8::one()));
    REQUIRE(m2.row_space_included(BMat8::one()));
    REQUIRE(m3.row_space_included(BMat8::one()));
}

TEST_AGREES2(BMat8Fixture, BMat8, row_space_included, row_space_included_ref,
             BMlist, "[BMat8][018]");
TEST_AGREES2(BMat8Fixture, BMat8, row_space_included, row_space_included_bitset,
             BMlist, "[BMat8][019]");

TEST_CASE_METHOD(BMat8Fixture, "BMat8::row_space_included2", "[BMat8][020]") {
    BMat8 a0 = BMat8::one();
    BMat8 b0 = BMat8(0);
    BMat8 a1 = BMat8(0);
    BMat8 b1 = BMat8::one();

    auto res = BMat8::row_space_included2(a0, b0, a1, b1);
    REQUIRE(res.first == a0.row_space_included(b0));
    REQUIRE(res.second == a1.row_space_included(b1));

    for (auto a0 : BMlist) {
        for (auto b0 : BMlist) {
            for (auto a1 : BMlist) {
                for (auto b1 : BMlist) {
                    auto res = BMat8::row_space_included2(a0, b0, a1, b1);
                    REQUIRE(res.first == a0.row_space_included(b0));
                    REQUIRE(res.second == a1.row_space_included(b1));
                }
            }
        }
    }
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::row_permuted", "[BMat8][021]") {
    REQUIRE(bm2.row_permuted(Perm16({1, 0})) == BMat8({{0, 1}, {1, 1}}));
    REQUIRE(bm2.row_permuted(Perm16({2, 1, 0})) ==
            BMat8({{0, 0, 0}, {0, 1, 0}, {1, 1, 0}}));
    REQUIRE(bm.row_permuted(Perm16({5, 3, 1, 4, 2, 0})) ==
            BMat8({{1, 1, 0, 0, 0, 0, 0, 1},
                   {1, 1, 0, 1, 1, 1, 1, 1},
                   {1, 1, 1, 1, 1, 1, 0, 1},
                   {0, 0, 1, 0, 0, 1, 1, 1},
                   {0, 1, 1, 1, 0, 1, 0, 1},
                   {0, 0, 0, 1, 0, 0, 1, 1},
                   {0, 1, 0, 0, 0, 0, 1, 1},
                   {0, 1, 1, 1, 1, 0, 1, 0}}));
    REQUIRE(BMat8::one().row_permuted(Perm16({5, 3, 1, 4, 2, 0})) ==
            BMat8({{0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::col_permuted", "[BMat8][022]") {
    REQUIRE(bm2.col_permuted(Perm16({1, 0})) == BMat8({{1, 1}, {1, 0}}));
    REQUIRE(bm2.col_permuted(Perm16({2, 1, 0})) ==
            BMat8({{0, 1, 1}, {0, 1, 0}, {0, 0, 0}}));
    REQUIRE(bm.col_permuted(Perm16({5, 3, 1, 4, 2, 0})) ==
            BMat8({{0, 1, 0, 0, 0, 0, 1, 1},
                   {1, 1, 1, 1, 1, 1, 0, 1},
                   {1, 1, 1, 0, 1, 0, 0, 1},
                   {1, 1, 1, 1, 0, 1, 1, 1},
                   {1, 0, 0, 0, 1, 0, 1, 1},
                   {0, 0, 1, 0, 0, 1, 0, 1},
                   {0, 0, 1, 0, 0, 0, 1, 1},
                   {0, 1, 1, 1, 1, 0, 1, 0}}));
    REQUIRE(BMat8::one().col_permuted(Perm16({4, 1, 3, 0, 2, 6, 5})) ==
            BMat8({{0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
}

TEST_CASE("BMat8::row_permutation_matrix", "[BMat8][023]") {
    REQUIRE(BMat8::row_permutation_matrix(Perm16({1, 0})) ==
            BMat8({{0, 1, 0, 0, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
    REQUIRE(BMat8::row_permutation_matrix(Perm16({1, 3, 4, 0, 2})) ==
            BMat8({{0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
    REQUIRE(BMat8::row_permutation_matrix(Perm16({5, 3, 1, 4, 2, 0})) ==
            BMat8({{0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
}

TEST_CASE("BMat8::col_permutation_matrix", "[BMat8][024]") {
    REQUIRE(BMat8::col_permutation_matrix(Perm16({1, 0})) ==
            BMat8({{0, 1, 0, 0, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
    REQUIRE(BMat8::col_permutation_matrix(Perm16({1, 3, 4, 0, 2})) ==
            BMat8({{0, 0, 0, 1, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
    REQUIRE(BMat8::col_permutation_matrix(Perm16({5, 3, 1, 4, 2, 0})) ==
            BMat8({{0, 0, 0, 0, 0, 1, 0, 0},
                   {0, 0, 1, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 1, 0, 0, 0},
                   {0, 1, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 1, 0, 0, 0, 0},
                   {1, 0, 0, 0, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0, 0, 1, 0},
                   {0, 0, 0, 0, 0, 0, 0, 1}}));
}

TEST_CASE_METHOD(BMat8Fixture, "BMat8::nr_rows", "[BMat8][025]") {
    REQUIRE(zero.nr_rows() == 0);
    REQUIRE(one1.nr_rows() == 1);
    REQUIRE(one2.nr_rows() == 2);
    REQUIRE(bm.nr_rows() == 8);
    REQUIRE(BMat8({{1, 0, 1}, {1, 1, 0}, {0, 0, 0}}).nr_rows() == 2);
}

// TEST_CASE("BMat8::right_perm_action_on_basis_ref", "[BMat8][026]") {
//     BMat8 m1({{1, 1, 0}, {1, 0, 1}, {0, 0, 0}});
//     BMat8 m2({{0, 0, 0}, {1, 0, 1}, {1, 1, 0}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16({1,0}));
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16({1,0}));
//
//     m1 = BMat8({{1, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}});
//     m2 = BMat8({{1, 0, 0, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16::one());
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16::one());
//
//     m1 = BMat8({{1, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}});
//     m2 = BMat8({{0, 0, 0, 0}, {1, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16::one());
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16::one());
//
//     m1 = BMat8({{0,1,0,0}, {0,0,1,0}, {1,0,0,1}, {0,0,0,0}});
//     m2 = BMat8({{1,0,0,1}, {0,0,1,0}, {0,1,0,0}, {0,0,0,1}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16({1,0}));
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16({1,0}));
//
//     m1 = BMat8({{0,0,0,1}, {1,0,0,0}, {0,0,1,0}, {0,1,0,0}});
//     m2 = BMat8({{0,1,0,0}, {0,0,1,0}, {1,0,0,0}, {0,0,0,1}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16({0,2,3,1}));
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16({0,2,3,1}));
//
//
//     m1 = BMat8({{0,0,0,1}, {0,0,1,0}, {0,1,0,0}, {1,0,0,0}});
//     m2 = BMat8({{0,1,0,0}, {0,0,0,1}, {1,0,0,0}, {0,0,1,0}});
//     REQUIRE(m1.right_perm_action_on_basis_ref(m2) == Perm16({2,0,3,1}));
//     REQUIRE(m1.right_perm_action_on_basis(m2) == Perm16({2,0,3,1}));
// }

}  // namespace HPCombi
