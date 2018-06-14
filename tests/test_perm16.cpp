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
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;

struct Fix {
  Fix() : zero(Vect16({}, 0)),
          P01(Vect16({0, 1}, 0)),
          P10(Vect16({1, 0}, 0)),
          P11(Vect16({1, 1}, 0)),
          P1(Vect16({}, 1)),
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
BOOST_AUTO_TEST_SUITE(Vect16_test)
//****************************************************************************//
BOOST_FIXTURE_TEST_CASE(IsPTransf, Fix) {
  BOOST_ASSERT(Fix::zero.is_partial_transformation());
  BOOST_ASSERT(Fix::P01.is_partial_transformation());
  BOOST_ASSERT(Fix::P10.is_partial_transformation());
  BOOST_ASSERT(not Vect16({16, 0}).is_partial_transformation());
  BOOST_ASSERT(Vect16({}, 0xff).is_partial_transformation());
  BOOST_ASSERT(Vect16({2, 0xff, 3}, 0).is_partial_transformation());

  BOOST_ASSERT(not Fix::zero.is_partial_transformation(15));
  BOOST_ASSERT(Fix::PPa.is_partial_transformation());
  BOOST_ASSERT(Fix::PPa.is_partial_transformation(6));
  BOOST_ASSERT(Fix::PPa.is_partial_transformation(5));
  BOOST_ASSERT(not Fix::PPa.is_partial_transformation(4));
  BOOST_ASSERT(not Fix::PPa.is_partial_transformation(1));
  BOOST_ASSERT(not Fix::PPa.is_partial_transformation(0));

  BOOST_ASSERT(Fix::RandT.is_partial_transformation());
  BOOST_ASSERT(Fix::RandT.is_partial_transformation(16));
  BOOST_ASSERT(not Fix::RandT.is_partial_transformation(15));
}

BOOST_FIXTURE_TEST_CASE(IsTransf, Fix) {
  BOOST_ASSERT(Fix::zero.is_transformation());
  BOOST_ASSERT(Fix::P01.is_transformation());
  BOOST_ASSERT(Fix::P10.is_transformation());
  BOOST_ASSERT(not Vect16({16, 0}).is_transformation());
  BOOST_ASSERT(not Vect16({}, 0xff).is_transformation());
  BOOST_ASSERT(not Vect16({2, 0xff, 3}, 0).is_transformation());

  BOOST_ASSERT(not Fix::zero.is_transformation(15));
  BOOST_ASSERT(Fix::PPa.is_transformation());
  BOOST_ASSERT(Fix::PPa.is_transformation(6));
  BOOST_ASSERT(Fix::PPa.is_transformation(5));
  BOOST_ASSERT(not Fix::PPa.is_transformation(4));
  BOOST_ASSERT(not Fix::PPa.is_transformation(1));
  BOOST_ASSERT(not Fix::PPa.is_transformation(0));

  BOOST_ASSERT(Fix::RandT.is_transformation());
  BOOST_ASSERT(Fix::RandT.is_transformation(16));
  BOOST_ASSERT(not Fix::RandT.is_transformation(15));
}

BOOST_FIXTURE_TEST_CASE(IsPerm, Fix) {
  BOOST_ASSERT(not Fix::zero.is_permutation());
  BOOST_ASSERT(not Fix::P01.is_permutation());
  BOOST_ASSERT(not Fix::P10.is_permutation());
  BOOST_ASSERT(not Vect16({16, 0}).is_permutation());
  BOOST_ASSERT(not Vect16({}, 0xff).is_permutation());
  BOOST_ASSERT(not Vect16({2, 0xff, 3}, 0).is_permutation());

  BOOST_ASSERT(not Fix::zero.is_permutation(15));
  BOOST_ASSERT(Fix::PPa.is_permutation());
  BOOST_ASSERT(Fix::PPa.is_permutation(6));
  BOOST_ASSERT(Fix::PPa.is_permutation(5));
  BOOST_ASSERT(not Fix::PPa.is_permutation(4));
  BOOST_ASSERT(not Fix::PPa.is_permutation(1));
  BOOST_ASSERT(not Fix::PPa.is_permutation(0));

  BOOST_ASSERT(Fix::RandT.is_permutation());
  BOOST_ASSERT(Fix::RandT.is_permutation(16));
  BOOST_ASSERT(not Fix::RandT.is_permutation(15));
}

BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//


//****************************************************************************//
BOOST_AUTO_TEST_SUITE(Transf16_test)
//****************************************************************************//

BOOST_FIXTURE_TEST_CASE(Transf16OperatorUInt64, Fix) {
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Transf16::one()), 0xf7e6d5c4b3a29180);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::zero), 0x0);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::P10), 0x1);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::P01), 0x100);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::P11), 0x101);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::P1), 0x1111111111111111);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::RandT), 0x9a854d7fce60b123);
}

BOOST_FIXTURE_TEST_CASE(Transf16ConstrUInt64, Fix) {
  BOOST_CHECK_EQUAL(static_cast<Transf16>(0x0), Fix::zero);
  BOOST_CHECK_EQUAL(static_cast<Transf16>(0x1), Fix::P10);
  BOOST_CHECK_EQUAL(static_cast<Transf16>(0x100), Fix::P01);
  for (auto p : Fix::Tlist)
    BOOST_CHECK_EQUAL(static_cast<Transf16>(static_cast<uint64_t>(p)), p);
}

BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//


//****************************************************************************//
BOOST_AUTO_TEST_SUITE(Perm16_test)
//****************************************************************************//

BOOST_FIXTURE_TEST_CASE(Perm16OperatorUInt64, Fix) {
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Perm16::one()), 0xf7e6d5c4b3a29180);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::PPa), 0xf7e6d5c0b4a39281);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::PPb), 0xd7e4f5c0b6a39281);
  BOOST_CHECK_EQUAL(static_cast<uint64_t>(Fix::RandPerm), 0x9a854d7fce60b123);

  for (auto p : { Perm16::one(), Fix::PPa, Fix::PPb, Fix::RandPerm })
    BOOST_CHECK_EQUAL(static_cast<Perm16>(static_cast<uint64_t>(p)), p);
}

BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//


BOOST_AUTO_TEST_CASE(Vect16TestEq) { BOOST_CHECK_EQUAL(Vect16(), Vect16()); }

BOOST_AUTO_TEST_CASE(Perm16TestEq) {
  BOOST_CHECK_EQUAL(Perm16::one() * Perm16::one(), Perm16::one());
}


BOOST_AUTO_TEST_CASE(PTransf16_image) {
  BOOST_CHECK_EQUAL(PTransf16({}).image(), 0xffff);
  BOOST_CHECK_EQUAL(PTransf16({4,4,4,4}).image(), 0xfff0);
  BOOST_CHECK_EQUAL(PTransf16({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}).image(), 0x02);
  BOOST_CHECK_EQUAL(PTransf16({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2}).image(), 0x04);
}

BOOST_AUTO_TEST_CASE(Vect16_remove_dups) {
  BOOST_CHECK_EQUAL(Vect16({}).remove_dups(), Vect16({}));
  BOOST_CHECK_EQUAL(Vect16({4,4,4,4}).remove_dups(),
                    Vect16({4,0,0,0}));
  BOOST_CHECK_EQUAL(Vect16({1,1,1,1,1,1,1,1,1,1,1,}).remove_dups(),
                    Vect16({1}));
  BOOST_CHECK_EQUAL(Vect16({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}).remove_dups(),
                    Vect16({1}));
  BOOST_CHECK_EQUAL(Vect16({2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2}).remove_dups(),
                    Vect16({2}));
  BOOST_CHECK_EQUAL(Vect16({2,2,3,3,3,1,5,5}).remove_dups(),
                    Vect16({2,0,3,0,0,1,5}));
}
