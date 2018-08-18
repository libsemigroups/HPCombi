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

#define BOOST_TEST_MODULE Perm16Tests

#include "perm16.hpp"
#include <boost/test/unit_test.hpp>

using HPCombi::epu8;
using HPCombi::Epu8;
using HPCombi::is_partial_transformation;
using HPCombi::is_transformation;
using HPCombi::is_permutation;

using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;

struct Fix {
    Fix() : zero(Epu8({}, 0)),
            P01(Epu8({0, 1}, 0)),
            P10(Epu8({1, 0}, 0)),
            P11(Epu8({1, 1}, 0)),
            P1(Epu8({}, 1)),
            RandT({3, 1, 0, 14, 15, 13, 5, 10, 2, 11, 6, 12, 7, 4, 8, 9}),
            PPa({1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            PPb({1, 2, 3, 6, 0, 5, 4, 7, 8, 9, 10, 11, 12, 15, 14, 13}),
            RandPerm(RandT),
            Tlist({zero, P01, P10, P11, P1, RandT, epu8(PPa), epu8(PPb)}) {
        BOOST_TEST_MESSAGE("setup fixture");
    }
    ~Fix() { BOOST_TEST_MESSAGE("teardown fixture"); }

    const Transf16 zero, P01, P10, P11, P1, RandT;
    const Perm16 PPa, PPb, RandPerm;
    const std::vector<Transf16> Tlist;

};


//****************************************************************************//
BOOST_AUTO_TEST_SUITE(Transf16_test)
//****************************************************************************//

BOOST_FIXTURE_TEST_CASE(Transf16OperatorUInt64, Fix) {
    BOOST_TEST(static_cast<uint64_t>(Transf16::one()) == 0xf7e6d5c4b3a29180);
    BOOST_TEST(static_cast<uint64_t>(zero) == 0x0);
    BOOST_TEST(static_cast<uint64_t>(P10) == 0x1);
    BOOST_TEST(static_cast<uint64_t>(P01) == 0x100);
    BOOST_TEST(static_cast<uint64_t>(P11) == 0x101);
    BOOST_TEST(static_cast<uint64_t>(P1) == 0x1111111111111111);
    BOOST_TEST(static_cast<uint64_t>(RandT) == 0x9a854d7fce60b123);
}

BOOST_FIXTURE_TEST_CASE(Transf16ConstrUInt64, Fix) {
    BOOST_TEST(static_cast<Transf16>(0x0) == zero);
    BOOST_TEST(static_cast<Transf16>(0x1) == P10);
    BOOST_TEST(static_cast<Transf16>(0x100) == P01);
    for (auto p : Tlist)
        BOOST_TEST(static_cast<Transf16>(static_cast<uint64_t>(p)) == p);
}

BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//


//****************************************************************************//
BOOST_AUTO_TEST_SUITE(Perm16_test)
//****************************************************************************//

BOOST_FIXTURE_TEST_CASE(Perm16OperatorUInt64, Fix) {
    BOOST_TEST(static_cast<uint64_t>(Perm16::one()) == 0xf7e6d5c4b3a29180);
    BOOST_TEST(static_cast<uint64_t>(PPa) == 0xf7e6d5c0b4a39281);
    BOOST_TEST(static_cast<uint64_t>(PPb) == 0xd7e4f5c0b6a39281);
    BOOST_TEST(static_cast<uint64_t>(RandPerm) == 0x9a854d7fce60b123);

    for (auto p : { Perm16::one(), PPa, PPb, RandPerm })
        BOOST_TEST(static_cast<Perm16>(static_cast<uint64_t>(p)) == p);
}

BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//


BOOST_AUTO_TEST_CASE(Perm16TestEq) {
    BOOST_TEST(Perm16::one() * Perm16::one() == Perm16::one());
}


BOOST_AUTO_TEST_CASE(PTransf16_image) {
    BOOST_TEST(PTransf16({}).image(), 0xffff);
    BOOST_TEST(PTransf16({4,4,4,4}).image(), 0xfff0);
    BOOST_TEST(PTransf16({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}).image(), 0x02);
    BOOST_TEST(PTransf16({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2}).image(), 0x04);
}
