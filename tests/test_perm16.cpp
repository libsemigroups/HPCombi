/******************************************************************************/
/*       Copyright (C) 2017 Florent Hivert <Florent.Hivert@lri.fr>,           */
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

#include "perm16.hpp"

#include "test_main.hpp"
#include <catch2/catch_test_macros.hpp>

namespace HPCombi {
const uint8_t FF = 0xff;

namespace {
std::vector<Perm16> all_perms(uint8_t sz) {
    std::vector<Perm16> res{};
    epu8 x = HPCombi::epu8id;
    res.push_back(x);
    auto &refx = HPCombi::as_array(x);
    while (std::next_permutation(refx.begin(), refx.begin() + sz)) {
        res.push_back(x);
    }
    return res;
};

std::vector<PPerm16> all_pperms(std::vector<Perm16> perms,
                                std::vector<epu8> masks) {
    std::vector<PPerm16> res{};
    for (epu8 mask : masks) {
        for (Perm16 p : perms) {
            res.push_back(p.v | mask);
        }
    }
    return res;
}

struct Perm16Fixture {
    Perm16Fixture()
        : zero(Epu8({}, 0)), P01(Epu8({0, 1}, 0)), P10(Epu8({1, 0}, 0)),
          P11(Epu8({1, 1}, 0)), P1(Epu8({}, 1)),
          RandT({3, 1, 0, 14, 15, 13, 5, 10, 2, 11, 6, 12, 7, 4, 8, 9}),
          PPa({1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
          PPb({1, 2, 3, 6, 0, 5, 4, 7, 8, 9, 10, 11, 12, 15, 14, 13}),
          RandPerm(RandT),
          Tlist({zero, P01, P10, P11, P1, RandT, epu8(PPa), epu8(PPb)}),
          PlistSmall(all_perms(6)), Plist(all_perms(9)),
          PPmasks({Epu8(0), Epu8(FF), Epu8({0}, FF), Epu8({0, 0}, FF),
                   Epu8({0, FF, 0}, FF), Epu8({0, FF, 0}, 0),
                   Epu8({0, FF, 0, FF, 0, 0, 0, FF, FF}, 0)}),
          PPlist(all_pperms(PlistSmall, PPmasks)) {}
    ~Perm16Fixture() = default;

    const Transf16 zero, P01, P10, P11, P1, RandT;
    const Perm16 PPa, PPb, RandPerm;
    const std::vector<Transf16> Tlist;
    const std::vector<Perm16> PlistSmall, Plist;
    const std::vector<epu8> PPmasks;
    const std::vector<PPerm16> PPlist;
};
}  // namespace

TEST_CASE("PTransf16::PTransf16", "[PTransf16][000]") {
    const uint8_t FF = 0xff;
    REQUIRE(PTransf16({}) == PTransf16::one());
    REQUIRE(PTransf16({0, 1, 2, 3}) == PTransf16::one());
    REQUIRE(PTransf16({1, 0}) == PTransf16({1, 0, 2}));
    REQUIRE(PTransf16({2}) == PTransf16({2, 1, 2}));
    REQUIRE(PTransf16({4, 5, 0}, {9, 0, 1}) ==
            PTransf16(
                {1, FF, FF, FF, 9, 0, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF}));
    REQUIRE(PTransf16({4, 5, 0, 8}, {9, 0, 1, 2}) ==
            PTransf16(
                {1, FF, FF, FF, 9, 0, FF, FF, 2, FF, FF, FF, FF, FF, FF, FF}));
    REQUIRE(PTransf16({4, 5, 0, 8}, {9, 0, 2, 2}) ==
            PTransf16(
                {2, FF, FF, FF, 9, 0, FF, FF, 2, FF, FF, FF, FF, FF, FF, FF}));
}

TEST_CASE("PTransf16::hash", "[PTransf16][001]") {
    REQUIRE(std::hash<PTransf16>()(PTransf16::one()) != 0);
    REQUIRE(std::hash<PTransf16>()(PTransf16(Epu8(1))) != 0);
    REQUIRE(std::hash<PTransf16>()(PTransf16({4, 5, 0}, {9, 0, 1})) != 0);
}

TEST_CASE("PTransf16::image_mask", "[PTransf16][002]") {
    REQUIRE_THAT(PTransf16({}).image_mask(), Equals(Epu8(FF)));
    REQUIRE_THAT(PTransf16({}).image_mask(false), Equals(Epu8(FF)));
    REQUIRE_THAT(PTransf16({}).image_mask(true), Equals(Epu8(0)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask(),
                 Equals(Epu8({0, 0, 0, 0}, FF)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask(false),
                 Equals(Epu8({0, 0, 0, 0}, FF)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask(true),
                 Equals(Epu8({FF, FF, FF, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8(1)).image_mask(), Equals(Epu8({0, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8(2)).image_mask(), Equals(Epu8({0, 0, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8({2, 2, 2, 0xf}, 2)).image_mask(),
                 Equals(Epu8({0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask(),
        Equals(Epu8({FF, 0, FF, 0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask(false),
        Equals(Epu8({FF, 0, FF, 0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask(true),
        Equals(Epu8({0, FF, 0, FF, FF, 0, FF, FF, FF, FF, FF, FF, FF, FF, FF, 0}, 0)));
}

TEST_CASE("PTransf16::image_mask_ref_ref", "[PTransf16][002]") {
    REQUIRE_THAT(PTransf16({}).image_mask_ref(), Equals(Epu8(FF)));
    REQUIRE_THAT(PTransf16({}).image_mask_ref(false), Equals(Epu8(FF)));
    REQUIRE_THAT(PTransf16({}).image_mask_ref(true), Equals(Epu8(0)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask_ref(),
                 Equals(Epu8({0, 0, 0, 0}, FF)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask_ref(false),
                 Equals(Epu8({0, 0, 0, 0}, FF)));
    REQUIRE_THAT(PTransf16({4, 4, 4, 4}).image_mask_ref(true),
                 Equals(Epu8({FF, FF, FF, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8(1)).image_mask_ref(), Equals(Epu8({0, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8(2)).image_mask_ref(), Equals(Epu8({0, 0, FF}, 0)));
    REQUIRE_THAT(PTransf16(Epu8({2, 2, 2, 0xf}, 2)).image_mask_ref(),
                 Equals(Epu8({0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask_ref(),
        Equals(Epu8({FF, 0, FF, 0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask_ref(false),
        Equals(Epu8({FF, 0, FF, 0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, FF}, 0)));
    REQUIRE_THAT(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).image_mask_ref(true),
        Equals(Epu8({0, FF, 0, FF, FF, 0, FF, FF, FF, FF, FF, FF, FF, FF, FF, 0}, 0)));
}

// TODO uncomment
// TEST_CASE("PTransf16::left_one", "[PTransf16][003]") {
//     REQUIRE(PTransf16({}).left_one() == PTransf16::one());
//     REQUIRE(PTransf16({4, 4, 4, 4}).left_one() ==
//                PTransf16({FF, FF, FF, FF}));
//     REQUIRE(PTransf16(Epu8(1)).left_one() == PTransf16(Epu8({FF, 1},
//     FF))); REQUIRE(PTransf16(Epu8(2)).left_one() ==
//                PTransf16(Epu8({FF, FF, 2}, FF)));
//     REQUIRE(PTransf16(Epu8({2, 2, 2, 0xf}, 2)).left_one() ==
//                PTransf16({FF, FF, 2, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF,
//                FF,
//                           FF, 15}));
//     REQUIRE(PTransf16(Epu8({FF, 2, 2, 0xf}, FF)).left_one() ==
//                PTransf16({FF, FF, 2, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF,
//                FF,
//                           FF, 15}));
//     REQUIRE(
//         PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).left_one() ==
//         PTransf16(
//             {0, FF, 2, FF, FF, 5, FF, FF, FF, FF, FF, FF, FF, FF, FF, 15 }));
//     REQUIRE(
//         PTransf16(Epu8({0, 2, FF, 0xf, 2, FF, 2, FF, 5}, FF)).left_one() ==
//         PTransf16(
//             {0, FF, 2, FF, FF, 5, FF, FF, FF, FF, FF, FF, FF, FF, FF, 15 }));
// }

TEST_CASE("PTransf16::domain_mask", "[PTransf16][004]") {
    REQUIRE(equal(PTransf16({}).domain_mask(), Epu8(FF)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).domain_mask(), Epu8(FF)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).domain_mask(false), Epu8(FF)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).domain_mask(true), Epu8(0)));
    REQUIRE(equal(PTransf16(Epu8(1)).domain_mask(), Epu8(FF)));
    REQUIRE(equal(PTransf16(Epu8(2)).domain_mask(), Epu8(FF)));
    REQUIRE(equal(PTransf16(Epu8({2, 2, 2, 0xf}, FF)).domain_mask(),
                  Epu8({FF, FF, FF, FF}, 0)));
    REQUIRE(equal(PTransf16(Epu8({FF, 2, 2, 0xf}, FF)).domain_mask(),
                  Epu8({0, FF, FF, FF}, 0)));
    REQUIRE(equal(
        PTransf16(Epu8({0, 2, FF, 0xf, 2, FF, 2, FF, 5}, FF)).domain_mask(),
        Epu8({FF, FF, 0, FF, FF, 0, FF, 0, FF}, 0)));
    REQUIRE(equal(PTransf16(Epu8({0, 2, FF, 0xf, 2, FF, 2, FF, 5}, FF))
                      .domain_mask(false),
                  Epu8({FF, FF, 0, FF, FF, 0, FF, 0, FF}, 0)));
    REQUIRE(equal(
        PTransf16(Epu8({0, 2, FF, 0xf, 2, FF, 2, FF, 5}, FF)).domain_mask(true),
        Epu8({0, 0, FF, 0, 0, FF, 0, FF, 0}, FF)));
}

TEST_CASE("PTransf16::right_one", "[PTransf16][005]") {
    REQUIRE(PTransf16({}).right_one() == PTransf16::one());
    REQUIRE(PTransf16({4, 4, 4, 4}).right_one() == PTransf16::one());
    REQUIRE(PTransf16(Epu8(1)).right_one() == PTransf16::one());
    REQUIRE(PTransf16(Epu8(2)).right_one() == PTransf16::one());
    REQUIRE(PTransf16(Epu8({2, 2, 2, 0xf}, FF)).right_one() ==
            PTransf16(Epu8({0, 1, 2, 3}, FF)));
    REQUIRE(PTransf16(Epu8({FF, 2, 2, 0xf}, FF)).right_one() ==
            PTransf16(
                {FF, 1, 2, 3, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF}));
    REQUIRE(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2)).right_one() ==
            PTransf16::one());
    REQUIRE(
        PTransf16(Epu8({0, 2, FF, 0xf, 2, FF, 2, FF, 5}, FF)).right_one() ==
        PTransf16({0, 1, FF, 3, 4, FF, 6, FF, 8, FF, FF, FF, FF, FF, FF, FF}));
}

TEST_CASE("PTransf16::rank_ref", "[PTransf16][006]") {
    REQUIRE(PTransf16({}).rank_ref() == 16);
    REQUIRE(PTransf16({4, 4, 4, 4}).rank_ref() == 12);
    REQUIRE(PTransf16({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                .rank_ref() == 1);
    REQUIRE(PTransf16({2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2})
                .rank_ref() == 1);
    REQUIRE(PTransf16({2, 2, 2, 0xf, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2})
                .rank_ref() == 2);
    REQUIRE(PTransf16({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2})
                .rank_ref() == 4);
    REQUIRE(PTransf16({1, 1, 1, FF, 1, 1, FF, 1, 1, FF, 1, FF, 1, 1, 1, 1})
                .rank_ref() == 1);
    REQUIRE(PTransf16({2, 2, 2, 2, 2, FF, 2, 2, 2, FF, 2, 2, 2, FF, 2, 2})
                .rank_ref() == 1);
    REQUIRE(PTransf16({2, 2, 2, 0xf, 2, FF, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2})
                .rank_ref() == 2);
    REQUIRE(PTransf16({0, 2, 2, 0xf, 2, 2, FF, 2, 5, 2, FF, 2, 2, 2, 2, 2})
                .rank_ref() == 4);
}

// TODO uncomment
// TEST_CASE("PTransf16::rank", "[PTransf16][007]") {
//     REQUIRE(PTransf16({}).rank() == 16);
//     REQUIRE(PTransf16({4, 4, 4, 4}).rank() == 12);
//     REQUIRE(
//         PTransf16({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}).rank() ==
//         1);
//     REQUIRE(
//         PTransf16({2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}).rank() ==
//         1);
//     REQUIRE(
//         PTransf16({2, 2, 2, 0xf, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}).rank()
//         == 2);
//     REQUIRE(
//         PTransf16({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2}).rank()
//         == 4);
// }

TEST_CASE("PTransf16::fix_points_mask", "[PTransf16][008]") {
    REQUIRE(equal(PTransf16({}).fix_points_mask(), Epu8(FF)));
    REQUIRE(equal(PTransf16({}).fix_points_mask(false), Epu8(FF)));
    REQUIRE(equal(PTransf16({}).fix_points_mask(true), Epu8(0)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).fix_points_mask(),
                  Epu8({0, 0, 0, 0}, FF)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).fix_points_mask(false),
                  Epu8({0, 0, 0, 0}, FF)));
    REQUIRE(equal(PTransf16({4, 4, 4, 4}).fix_points_mask(true),
                  Epu8({FF, FF, FF, FF}, 0)));
    REQUIRE(equal(PTransf16(Epu8(1)).fix_points_mask(), Epu8({0, FF}, 0)));
    REQUIRE(equal(PTransf16(Epu8(2)).fix_points_mask(), Epu8({0, 0, FF}, 0)));
    REQUIRE(equal(PTransf16(Epu8({2, 2, 2, 0xf}, 7)).fix_points_mask(),
                  Epu8({0, 0, FF, 0, 0, 0, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0}, 0)));
    REQUIRE(equal(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 14, 5, 2}, 2)).fix_points_mask(),
        Epu8({FF, 0, FF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0)));
    REQUIRE(
        equal(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 8, 2}, 14))
                  .fix_points_mask(false),
              Epu8({FF, 0, FF, 0, 0, 0, 0, 0, FF, 0, 0, 0, 0, 0, FF, 0}, 0)));
    REQUIRE(equal(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2))
                      .fix_points_mask(true),
                  Epu8({0, FF, 0}, FF)));
}

TEST_CASE("PTransf16::fix_points_bitset", "[PTransf16][009]") {
    REQUIRE(PTransf16({}).fix_points_bitset() == 0xFFFF);
    REQUIRE(PTransf16({}).fix_points_bitset(false) == 0xFFFF);
    REQUIRE(PTransf16({}).fix_points_bitset(true) == 0);
    REQUIRE(PTransf16({4, 4, 4, 4}).fix_points_bitset() == 0xFFF0);
    REQUIRE(PTransf16({4, 4, 4, 4}).fix_points_bitset(false) == 0xFFF0);
    REQUIRE(PTransf16({4, 4, 4, 4}).fix_points_bitset(true) == 0x000F);
    REQUIRE(PTransf16(Epu8(1)).fix_points_bitset() == 0x0002);
    REQUIRE(PTransf16(Epu8(2)).fix_points_bitset() == 0x0004);
    REQUIRE(PTransf16(Epu8({2, 2, 2, 0xf}, 7)).fix_points_bitset() == 0x0084);
    REQUIRE(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 14, 5, 2}, 2))
                .fix_points_bitset() == 0x5);
    REQUIRE(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 8, 2}, 14))
                .fix_points_bitset(false) == 0x4105);
    REQUIRE(PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 5, 2}, 2))
                .fix_points_bitset(true) == 0xFFFA);
}

TEST_CASE("PTransf16::nb_fix_points", "[PTransf16][010]") {
    REQUIRE(PTransf16({}).nb_fix_points() == 16);
    REQUIRE(PTransf16({4, 4, 4, 4}).nb_fix_points() == 12);
    REQUIRE(PTransf16(Epu8(1)).nb_fix_points() == 1);
    REQUIRE(PTransf16(Epu8(2)).nb_fix_points() == 1);
    REQUIRE(PTransf16(Epu8({2, 2, 2, 0xf}, 7)).nb_fix_points() == 2);
    REQUIRE(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 14, 5, 2}, 2)).nb_fix_points() ==
        2);
    REQUIRE(
        PTransf16(Epu8({0, 2, 2, 0xf, 2, 2, 2, 2, 8, 2}, 14)).nb_fix_points() ==
        4);
}

TEST_CASE_METHOD(Perm16Fixture, "Transf16::operator uint64",
                 "[Transf16][011]") {
    REQUIRE(static_cast<uint64_t>(Transf16::one()) == 0xf7e6d5c4b3a29180);
    REQUIRE(static_cast<uint64_t>(zero) == 0x0);
    REQUIRE(static_cast<uint64_t>(P10) == 0x1);
    REQUIRE(static_cast<uint64_t>(P01) == 0x100);
    REQUIRE(static_cast<uint64_t>(P11) == 0x101);
    REQUIRE(static_cast<uint64_t>(P1) == 0x1111111111111111);
    REQUIRE(static_cast<uint64_t>(RandT) == 0x9a854d7fce60b123);
}

TEST_CASE_METHOD(Perm16Fixture, "Transf16::Transf16(uint64_t)",
                 "[Transf16][012]") {
    REQUIRE(static_cast<Transf16>(0x0) == zero);
    REQUIRE(static_cast<Transf16>(0x1) == P10);
    REQUIRE(static_cast<Transf16>(0x100) == P01);
    for (auto p : Tlist) {
        REQUIRE(static_cast<Transf16>(static_cast<uint64_t>(p)) == p);
    }
}

TEST_CASE_METHOD(Perm16Fixture, "Transf16::hash", "[Transf16][013]") {
    REQUIRE(std::hash<Transf16>()(Transf16::one()) != 0);
    REQUIRE(std::hash<Transf16>()(Transf16(Epu8(1))) != 0);
    REQUIRE(std::hash<Transf16>()(RandT) != 0);
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::operator uint64_t", "[Perm16][014]") {
    REQUIRE(static_cast<uint64_t>(Perm16::one()) == 0xf7e6d5c4b3a29180);
    REQUIRE(static_cast<uint64_t>(PPa) == 0xf7e6d5c0b4a39281);
    REQUIRE(static_cast<uint64_t>(PPb) == 0xd7e4f5c0b6a39281);
    REQUIRE(static_cast<uint64_t>(RandPerm) == 0x9a854d7fce60b123);

    for (auto p : {Perm16::one(), PPa, PPb, RandPerm}) {
        REQUIRE(static_cast<Perm16>(static_cast<uint64_t>(p)) == p);
    }
}

TEST_CASE("Perm::operator==", "[Perm16][015]") {
    REQUIRE(Perm16::one() * Perm16::one() == Perm16::one());
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::hash", "[Perm16][016]") {
    REQUIRE(std::hash<Perm16>()(Transf16::one()) != 0);
    REQUIRE(std::hash<Perm16>()(PPa) != 0);
    REQUIRE(std::hash<Perm16>()(RandPerm) != 0);
}

TEST_CASE("PPerm16::PPerm16", "[PPerm16][017]") {
    const uint8_t FF = 0xff;
    REQUIRE(
        PPerm16({4, 5, 0}, {9, 0, 1}) ==
        PPerm16({1, FF, FF, FF, 9, 0, FF, FF, FF, FF, FF, FF, FF, FF, FF, FF}));
    REQUIRE(
        PPerm16({4, 5, 0, 8}, {9, 0, 1, 2}) ==
        PPerm16({1, FF, FF, FF, 9, 0, FF, FF, 2, FF, FF, FF, FF, FF, FF, FF}));
}

TEST_CASE("PPerm16::hash", "[PPerm16][018]") {
    REQUIRE(std::hash<PPerm16>()(PPerm16::one()) != 0);
    REQUIRE(std::hash<PPerm16>()(PPerm16({4, 5, 0}, {9, 0, 1})) != 0);
}
// TODO uncomment
// TEST_CASE_METHOD(Perm16Fixture, "PPerm16::left_one", "[PPerm16][019]") {
//     REQUIRE(PPerm16({}).left_one() == PPerm16::one());
//     REQUIRE(PPerm16({FF, FF, FF, 4}).left_one() ==
//                PPerm16({FF, FF, FF, FF}));
//     REQUIRE(PPerm16({FF, 4, FF, FF}).left_one() ==
//                PPerm16({FF, FF, FF, FF}));
//     for (auto pp : PPlist) {
//         REQUIRE(pp.left_one() * pp == pp);
//     }
// }

TEST_CASE_METHOD(Perm16Fixture, "PPerm16::right_one", "[PPerm16][020]") {
    REQUIRE(PPerm16({}).right_one() == PPerm16::one());
    REQUIRE(PPerm16({FF, FF, FF, 4}).right_one() == PPerm16({FF, FF, FF}));
    REQUIRE(PPerm16({FF, 4, FF, FF}).right_one() == PPerm16({FF, 1, FF, FF}));
    for (auto pp : PPlist) {
        REQUIRE(pp * pp.right_one() == pp);
    }
}

#ifdef SIMDE_X86_SSE4_2_NATIVE
TEST_AGREES(Perm16Fixture, PPerm16, inverse_ref, inverse_find, PPlist,
            "[PPerm16][021]");
#endif

TEST_CASE_METHOD(Perm16Fixture, "Perm16::fix_points_mask", "[PPerm16][022]") {
    REQUIRE(equal(PTransf16::one().fix_points_mask(), Epu8(FF)));
    REQUIRE(equal(Perm16::one().fix_points_mask(), Epu8(FF)));
    REQUIRE(equal(PPa.fix_points_mask(), Epu8({0, 0, 0, 0, 0}, FF)));
    REQUIRE(equal(PPb.fix_points_mask(), (epu8{0, 0, 0, 0, 0, FF, 0, FF, FF, FF,
                                               FF, FF, FF, 0, FF, 0})));
    REQUIRE(equal(RandPerm.fix_points_mask(), Epu8({0, FF}, 0)));

    REQUIRE(equal(Perm16::one().fix_points_mask(false), Epu8(FF)));
    REQUIRE(equal(PPa.fix_points_mask(false), Epu8({0, 0, 0, 0, 0}, FF)));
    REQUIRE(
        equal(PPb.fix_points_mask(false),
              (epu8{0, 0, 0, 0, 0, FF, 0, FF, FF, FF, FF, FF, FF, 0, FF, 0})));
    REQUIRE(equal(RandPerm.fix_points_mask(false), Epu8({0, FF}, 0)));

    REQUIRE(equal(Perm16::one().fix_points_mask(true), Epu8(0)));
    REQUIRE(equal(PPa.fix_points_mask(true), Epu8({FF, FF, FF, FF, FF}, 0)));
    REQUIRE(equal(PPb.fix_points_mask(true), (epu8{FF, FF, FF, FF, FF, 0, FF, 0,
                                                   0, 0, 0, 0, 0, FF, 0, FF})));
    REQUIRE(equal(RandPerm.fix_points_mask(true), Epu8({FF, 0}, FF)));
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::smallest_fix_point", "[Perm16][023]") {
    REQUIRE(Perm16::one().smallest_fix_point() == 0);
    REQUIRE(PPa.smallest_fix_point() == 5);
    REQUIRE(PPb.smallest_fix_point() == 5);
    REQUIRE(RandPerm.smallest_fix_point() == 1);
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::smallest_moved_point",
                 "[Perm16][024]") {
    REQUIRE(Perm16::one().smallest_moved_point() == FF);
    REQUIRE(PPa.smallest_moved_point() == 0);
    REQUIRE(PPb.smallest_moved_point() == 0);
    REQUIRE(RandPerm.smallest_moved_point() == 0);
    REQUIRE(Perm16({0, 1, 3, 2}).smallest_moved_point() == 2);
}

// TODO broken test
// TEST_CASE_METHOD(Perm16Fixture, "Perm16::largest_fix_point", "[Perm16][025]")
// {
//     REQUIRE(Perm16::one().largest_fix_point() == 15);
//     REQUIRE(PPa.largest_fix_point() == 15);
//     REQUIRE(PPb.largest_fix_point() == 14);
//     REQUIRE(RandPerm.largest_fix_point() == 1);
// }

TEST_CASE_METHOD(Perm16Fixture, "Perm16::nb_fix_points", "[Perm16][026]") {
    REQUIRE(Perm16::one().nb_fix_points() == 16);
    REQUIRE(PPa.nb_fix_points() == 11);
    REQUIRE(PPb.nb_fix_points() == 8);
    REQUIRE(RandPerm.nb_fix_points() == 1);
    REQUIRE(Perm16({0, 1, 3, 2}).nb_fix_points() == 14);
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::inverse_ref", "[Perm16][027]") {
    REQUIRE(PPa * PPa.inverse() == Perm16::one());
    REQUIRE(PPa.inverse() * PPa == Perm16::one());
    REQUIRE(PPb * PPb.inverse() == Perm16::one());
    REQUIRE(PPb.inverse() * PPb == Perm16::one());
    REQUIRE(RandPerm * RandPerm.inverse() == Perm16::one());
    REQUIRE(RandPerm.inverse() * RandPerm == Perm16::one());

    for (Perm16 p : Plist) {
        REQUIRE(p * p.inverse() == Perm16::one());
        REQUIRE(p.inverse() * p == Perm16::one());
    }
}

TEST_AGREES(Perm16Fixture, Perm16, inverse_ref, inverse_find, Plist,
            "[Perm16][028]");
TEST_AGREES(Perm16Fixture, Perm16, inverse_ref, inverse_pow, Plist,
            "[Perm16][029]");
TEST_AGREES(Perm16Fixture, Perm16, inverse_ref, inverse_cycl, Plist,
            "[Perm16][030]");
TEST_AGREES(Perm16Fixture, Perm16, inverse_ref, inverse, Plist,
            "[Perm16][031]");

TEST_CASE_METHOD(Perm16Fixture, "Perm16::lehmer_ref", "[Perm16][032]") {
    REQUIRE(equal(Perm16::one().lehmer(), zero));
    REQUIRE(equal(PPa.lehmer(),
                  (epu8{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));
    REQUIRE(equal(PPb.lehmer(),
                  (epu8{1, 1, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0})));
}

TEST_AGREES_EPU8(Perm16Fixture, Perm16, lehmer_ref, lehmer_arr, Plist,
                 "[Perm16][033]");
TEST_AGREES_EPU8(Perm16Fixture, Perm16, lehmer_ref, lehmer, Plist,
                 "[Perm16][034]");

TEST_CASE_METHOD(Perm16Fixture, "Perm16::length_ref", "[Perm16][035]") {
    REQUIRE(Perm16::one().length() == 0);
    REQUIRE(PPa.length() == 4);
    REQUIRE(PPb.length() == 10);
}

TEST_AGREES(Perm16Fixture, Perm16, length_ref, length_arr, Plist,
            "[Perm16][036]");
TEST_AGREES(Perm16Fixture, Perm16, length_ref, length, Plist, "[Perm16][037]");

TEST_CASE_METHOD(Perm16Fixture, "Perm16::nb_descents_ref", "[Perm16][038]") {
    REQUIRE(Perm16::one().nb_descents_ref() == 0);
    REQUIRE(PPa.nb_descents_ref() == 1);
    REQUIRE(PPb.nb_descents_ref() == 4);
    REQUIRE(Perm16::one().nb_descents() == 0);
}

TEST_AGREES(Perm16Fixture, Perm16, nb_descents_ref, nb_descents, Plist,
            "[Perm16][039]");

TEST_CASE_METHOD(Perm16Fixture, "Perm16::nb_cycles_ref", "[Perm16][040]") {
    REQUIRE(Perm16::one().nb_cycles_ref() == 16);
    REQUIRE(PPa.nb_cycles_ref() == 12);
    REQUIRE(PPb.nb_cycles_ref() == 10);
}

TEST_AGREES(Perm16Fixture, Perm16, nb_cycles_ref, nb_cycles, Plist,
            "[Perm16][041]");

TEST_CASE_METHOD(Perm16Fixture, "Perm16::left_weak_leq_ref", "[Perm16][042]") {
    REQUIRE(Perm16::one().left_weak_leq_ref(Perm16::one()));
    REQUIRE(Perm16::one().left_weak_leq_ref(PPa));
    REQUIRE(Perm16::one().left_weak_leq_ref(PPb));
    REQUIRE(PPa.left_weak_leq_ref(PPa));
    REQUIRE(PPb.left_weak_leq_ref(PPb));
}

TEST_CASE_METHOD(Perm16Fixture, "Perm16::left_weak_leq", "[Perm16][043]") {
    for (auto u : PlistSmall) {
        for (auto v : PlistSmall) {
            REQUIRE(u.left_weak_leq(v) == u.left_weak_leq_ref(v));
            REQUIRE(u.left_weak_leq_length(v) == u.left_weak_leq_ref(v));
        }
    }
}
}  // namespace HPCombi
