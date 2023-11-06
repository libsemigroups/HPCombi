////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////

#ifndef HPCOMBI_TESTS_TEST_MAIN_HPP_
#define HPCOMBI_TESTS_TEST_MAIN_HPP_

#include <string>

#include "hpcombi/epu8.hpp"
#include "hpcombi/xpu8.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#define TEST_AGREES(fixture, ref, fun, vct, tags)                              \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p : vct) {                                                   \
            CHECK(p.fun() == p.ref());                                         \
        }                                                                      \
    }

#define TEST_AGREES_FUN(fixture, ref, fun, vct, tags)                          \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p : vct) {                                                   \
            CHECK(fun(p) == ref(p));                                           \
        }                                                                      \
    }

#define TEST_AGREES2(fixture, ref, fun, vct, tags)                             \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p1 : vct) {                                                  \
            for (auto p2 : vct) {                                              \
                CHECK(p1.fun(p2) == p1.ref(p2));                               \
            }                                                                  \
        }                                                                      \
    }

#define TEST_AGREES2_FUN(fixture, ref, fun, vct, tags)                         \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p1 : vct) {                                                  \
            for (auto p2 : vct) {                                              \
                CHECK(fun(p1, p2) == ref(p1, p2));                             \
            }                                                                  \
        }                                                                      \
    }

#define TEST_AGREES_EPU8(fixture, ref, fun, vct, tags)                         \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p : vct) {                                                   \
            CHECK_THAT(p.fun(), Equals(p.ref()));                              \
        }                                                                      \
    }

#define TEST_AGREES_FUN_EPU8(fixture, ref, fun, vct, tags)                     \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p : vct) {                                                   \
            CHECK_THAT(fun(p), Equals(ref(p)));                                \
        }                                                                      \
    }

#define TEST_AGREES2_EPU8(fixture, ref, fun, vct, tags)                        \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p1 : vct) {                                                  \
            for (auto p2 : vct) {                                              \
                CHECK_THAT(p1.fun(p2), Equals(p1.ref(p2)));                    \
            }                                                                  \
        }                                                                      \
    }

#define TEST_AGREES2_FUN_EPU8(fixture, ref, fun, vct, tags)                    \
    TEST_CASE_METHOD(fixture, #ref " == " #fun, tags) {                        \
        for (auto p1 : vct) {                                                  \
            for (auto p2 : vct) {                                              \
                CHECK_THAT(fun(p1, p2), Equals(ref(p1, p2)));                  \
            }                                                                  \
        }                                                                      \
    }

template <typename TPU>
struct Equals : Catch::Matchers::MatcherGenericBase {
    Equals(TPU v) : v(v) {}

    bool match(TPU w) const { return HPCombi::equal(v, w); }

    std::string describe() const override {
        return "\n!=\n" + std::to_string(v);
    }

  private:
    const TPU v;
};

#endif  // HPCOMBI_TESTS_TEST_MAIN_HPP_
