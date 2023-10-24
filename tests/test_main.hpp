//
// libsemigroups - C++ library for semigroups and monoids
// Copyright (C) 2019 James D. Mitchell
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef HPCOMBI_TESTS_TEST_MAIN_HPP_
#define HPCOMBI_TESTS_TEST_MAIN_HPP_

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

#endif  // HPCOMBI_TESTS_TEST_MAIN_HPP_
