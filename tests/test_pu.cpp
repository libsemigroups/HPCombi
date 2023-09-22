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

#define BOOST_TEST_MODULE EPUTests

#include <boost/test/unit_test.hpp>
#include <boost/functional.hpp>
#include <vector>

#include "epu.hpp"
#include <iostream>

using namespace HPCombi;

#define EPU_EQUAL(p1, p2)  BOOST_CHECK_PREDICATE(equal, (p1)(p2))
#define EPU_NOT_EQUAL(p1, p2)  BOOST_CHECK_PREDICATE(boost::not2(equal), (p1)(p2))

#define TEST_AGREES(ref, fun) \
    BOOST_FIXTURE_TEST_CASE(EPU_agrees_##fun, Fix) { \
        for (auto x : v)  BOOST_TEST(fun(x) == ref(x)); \
    }
#define TEST_PU8_AGREES(ref, fun) \
    BOOST_FIXTURE_TEST_CASE(EPU_agrees_##fun, Fix) { \
        for (auto x : v)  PU8_EQUAL(fun(x), ref(x));        \
    }


struct Fix {
    Fix() : zero(Pu8({}, 0)), P01(Pu8({0, 1}, 0)),
            P10(Pu8({1, 0}, 0)), P11(Pu8({1, 1}, 0)),
            P1(Pu8({}, 1)),
            P112(Pu8({1, 1}, 2)),
            Pa(pu8{ 1, 2, 3, 4, 0, 5, 6, 7, 8}),
            Pb(pu8{ 1, 2, 3, 6, 0, 5, 4, 7, 8}),
            RP(pu8{ 3, 1, 0,14,15,13, 5,10, 2}),
            Pa1(Pu8({4, 2, 5, 1, 2, 7}, 1)),
            Pa2(Pu8({4, 2, 5, 1, 2, 9}, 1)),
            P51(Pu8({5,1}, 6)),
            // Elements should be sorted in alphabetic order here
            v({zero, P01, pu8id, P10, P11, P1, P112, Pa, Pb, RP,
               Pa1, Pa2, P51}),
        {
            BOOST_TEST_MESSAGE("setup fixture");
        }
    ~Fix() { BOOST_TEST_MESSAGE("teardown fixture"); }

    const pu8 zero, P01, P10, P11, P1, P112, Pa, Pb, RP,
        Pa1, Pa2, P51;
    const std::vector<pu8> v;
    const std::array<uint8_t, 8> av;
};


//****************************************************************************//
BOOST_AUTO_TEST_SUITE(PU8_compare)
//****************************************************************************//
BOOST_FIXTURE_TEST_CASE(PU8_equal, Fix) {
    BOOST_TEST(equal(Pc, Pc));
    BOOST_TEST_FALSE(equal(zero, P01));
}
//****************************************************************************//
BOOST_AUTO_TEST_SUITE_END()
//****************************************************************************//
