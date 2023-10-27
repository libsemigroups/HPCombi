//****************************************************************************//
//       Copyright (C) 2023 James D. Mitchell <jdm3@st-andrews.ac.uk>         //
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
//****************************************************************************//

#ifndef HPCOMBI_TESTS_TEST_MAIN_HPP_
#define HPCOMBI_TESTS_TEST_MAIN_HPP_

#include <string>

#include "epu.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#define TEST_AGREES(fixture, type, ref, fun, vct, tags)                        \
    TEST_CASE_METHOD(fixture, #type "::" #ref " == " #type "::" #fun, tags) {  \
        for (type p : vct) {                                                   \
            REQUIRE(p.fun() == p.ref());                                       \
        }                                                                      \
    }

#define TEST_AGREES2(fixture, type, ref, fun, vct, tags)                       \
    TEST_CASE_METHOD(fixture, #type "::" #ref " == " #type "::" #fun, tags) {  \
        for (type p1 : vct) {                                                  \
            for (type p2 : vct) {                                              \
                REQUIRE(p1.fun(p2) == p1.ref(p2));                             \
            }                                                                  \
        }                                                                      \
    }

#define TEST_AGREES_EPU8(fixture, type, ref, fun, vct, tags)                   \
    TEST_CASE_METHOD(fixture, #type "::" #ref " == " #type "::" #fun, tags) {  \
        for (type p : vct) {                                                   \
            REQUIRE(equal(p.fun(), p.ref()));                                  \
        }                                                                      \
    }


struct Equals : Catch::Matchers::MatcherGenericBase {
    Equals(HPCombi::epu8 v) : v(v) {}

    bool match(HPCombi::epu8 w) const { return HPCombi::equal(v, w); }

    std::string describe() const override {
      return "\n!=\n" + std::to_string(v);
    }

private:

    const HPCombi::epu8 v;
};

#endif  // HPCOMBI_TESTS_TEST_MAIN_HPP_
