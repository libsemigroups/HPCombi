////////////////////////////////////////////////////////////////////////////////
//     Copyright (C) 2016-2023 Florent Hivert <Florent.Hivert@lri.fr>,        //
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

#include <iostream>
#include <vector>

#include "test_main.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

#include "hpcombi/xpu8.hpp"

namespace HPCombi {

// auto IsSorted =
//    Catch::Matchers::Predicate<xpu8>(is_sorted, "is_sorted");

struct Fix {
    Fix()
        : zero(Xpu8({}, 0)), P01(Xpu8({0, 1}, 0)), P10(Xpu8({1, 0}, 0)),
          P11(Xpu8({1, 1}, 0)), P1(Xpu8({}, 1)), P112(Xpu8({1, 1}, 2)),
          Pa(xpu8{1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
          Pb(xpu8{1, 2, 3, 6, 0, 5, 4, 7, 8, 9, 10, 11, 12, 15, 14, 13}),
          RP(xpu8{3, 1, 0, 14, 15, 13, 5, 10, 2, 11, 6, 12, 7, 4, 8, 9}),
          Pa1(Xpu8({4, 2, 5, 1, 2, 7, 7, 3, 4, 2}, 1)),
          Pa2(Xpu8({4, 2, 5, 1, 2, 9, 7, 3, 4, 2}, 1)), P51(Xpu8({5, 1}, 6)),
          Pv(xpu8{5, 5, 2, 5, 1, 6, 12, 4, 0, 3, 2, 11, 12, 13, 14, 15}),
          Pw(xpu8{5, 5, 2, 9, 1, 6, 12, 4, 0, 4, 4, 4, 12, 13, 14, 15}),
          P5(Xpu8({}, 5)), Pc(Xpu8({23, 5, 21, 5, 43, 36}, 7)),
          // Elements should be sorted in alphabetic order here
          v({zero, P01, Xpu8.id(), P10, P11, P1, P112, Pa, Pb, RP, Pa1, Pa2, P51,
             Pv, Pw, P5, Xpu8.rev(), Pc}),
          av({{5, 5, 2, 5, 1, 6, 12, 4, 0, 3, 2, 11, 12, 13, 14, 15}}) {}
    ~Fix() = default;

    const xpu8 zero, P01, P10, P11, P1, P112, Pa, Pb, RP, Pa1, Pa2, P51, Pv, Pw,
        P5, Pc;
    const std::vector<xpu8> v;
    const std::array<uint8_t, 16> av;
};

// TEST_CASE_METHOD(Fix, "Xpu8::first_diff_ref", "[Xpu8][000]") {
//     CHECK(first_diff_ref(Pc, Pc) == 16);
//     CHECK(first_diff_ref(zero, P01) == 1);
//     CHECK(first_diff_ref(zero, P10) == 0);
//     CHECK(first_diff_ref(zero, P01, 1) == 16);
//     CHECK(first_diff_ref(zero, P01, 2) == 1);
//     CHECK(first_diff_ref(Pa1, Pa2, 2) == 16);
//     CHECK(first_diff_ref(Pa1, Pa2, 4) == 16);
//     CHECK(first_diff_ref(Pa1, Pa2, 5) == 16);
//     CHECK(first_diff_ref(Pa1, Pa2, 6) == 5);
//     CHECK(first_diff_ref(Pa1, Pa2, 7) == 5);
//     CHECK(first_diff_ref(Pa1, Pa2) == 5);
//     CHECK(first_diff(Pv, Pw) == 3);
//     for (int i = 0; i < 16; i++)
//         CHECK(first_diff(Pv, Pw, i) == (i <= 3 ? 16 : 3));
// }

// #ifdef SIMDE_X86_SSE4_2_NATIVE
// TEST_CASE_METHOD(Fix, "Xpu8::first_diff_cmpstr", "[Xpu8][001]") {
//     for (auto x : v) {
//         for (auto y : v) {
//             CHECK(first_diff_cmpstr(x, y) == first_diff_ref(x, y));
//             for (int i = 0; i < 17; i++)
//                 CHECK(first_diff_cmpstr(x, y, i) == first_diff_ref(x, y, i));
//         }
//     }
// }
// #endif
// TEST_CASE_METHOD(Fix, "Xpu8::first_diff_mask", "[Xpu8][002]") {
//     for (auto x : v) {
//         for (auto y : v) {
//             CHECK(first_diff_mask(x, y) == first_diff_ref(x, y));
//             for (int i = 0; i < 17; i++)
//                 CHECK(first_diff_mask(x, y, i) == first_diff_ref(x, y, i));
//         }
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::last_diff_ref", "[Xpu8][003]") {
//     CHECK(last_diff_ref(Pc, Pc) == 16);
//     CHECK(last_diff_ref(zero, P01) == 1);
//     CHECK(last_diff_ref(zero, P10) == 0);
//     CHECK(last_diff_ref(zero, P01, 1) == 16);
//     CHECK(last_diff_ref(zero, P01, 2) == 1);
//     CHECK(last_diff_ref(P1, Pa1) == 9);
//     CHECK(last_diff_ref(P1, Pa1, 12) == 9);
//     CHECK(last_diff_ref(P1, Pa1, 9) == 8);
//     CHECK(last_diff_ref(Pa1, Pa2, 2) == 16);
//     CHECK(last_diff_ref(Pa1, Pa2, 4) == 16);
//     CHECK(last_diff_ref(Pa1, Pa2, 5) == 16);
//     CHECK(last_diff_ref(Pa1, Pa2, 6) == 5);
//     CHECK(last_diff_ref(Pa1, Pa2, 7) == 5);
//     CHECK(last_diff_ref(Pa1, Pa2) == 5);
//     const std::array<uint8_t, 17> res{
//         {16, 16, 16, 16, 3, 3, 3, 3, 3, 3, 9, 10, 11, 11, 11, 11, 11}};
//     for (int i = 0; i <= 16; i++) {
//         CHECK(last_diff_ref(Pv, Pw, i) == res[i]);
//     }
// }
// #ifdef SIMDE_X86_SSE4_2_NATIVE
// TEST_CASE_METHOD(Fix, "Xpu8::last_diff_cmpstr", "[Xpu8][004]") {
//     for (auto x : v) {
//         for (auto y : v) {
//             CHECK(last_diff_cmpstr(x, y) == last_diff_ref(x, y));
//             for (int i = 0; i < 17; i++)
//                 CHECK(last_diff_cmpstr(x, y, i) == last_diff_ref(x, y, i));
//         }
//     }
// }
// #endif

// TEST_CASE_METHOD(Fix, "Xpu8::last_diff_mask", "[Xpu8][005]") {
//     for (auto x : v) {
//         for (auto y : v) {
//             CHECK(last_diff_mask(x, y) == last_diff_ref(x, y));
//             for (int i = 0; i < 17; i++)
//                 CHECK(last_diff_mask(x, y, i) == last_diff_ref(x, y, i));
//         }
//     }
// }

TEST_CASE_METHOD(Fix, "Xpu8::is_all_zero", "[Xpu8][006]") {
    CHECK(is_all_zero(zero));
    for (size_t i = 1; i < v.size(); i++) {
        CHECK(!is_all_zero(v[i]));
    }
}

TEST_CASE_METHOD(Fix, "Xpu8::is_all_one", "[Xpu8][007]") {
    for (size_t i = 0; i < v.size(); i++) {
        CHECK(!is_all_one(v[i]));
    }
    CHECK(is_all_one(Xpu8(0xFF)));
}

TEST_CASE_METHOD(Fix, "Xpu8::equal", "[Xpu8][008]") {
    for (size_t i = 0; i < v.size(); i++) {
        xpu8 a = v[i];
        for (size_t j = 0; j < v.size(); j++) {
            xpu8 b = v[j];
            if (i == j) {
                CHECK(equal(a, b));
                CHECK(!not_equal(a, b));
                CHECK(std::equal_to<xpu8>()(a, b));
                CHECK(!std::not_equal_to<xpu8>()(a, b));
            } else {
                CHECK(!equal(a, b));
                CHECK(not_equal(a, b));
                CHECK(std::not_equal_to<xpu8>()(a, b));
                CHECK(!std::equal_to<xpu8>()(a, b));
            }
        }
    }
}

TEST_CASE_METHOD(Fix, "Xpu8::not_equal", "[Xpu8][009]") {
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < v.size(); j++) {
            if (i == j) {
                CHECK(!not_equal(v[i], v[j]));
            } else {
                CHECK(not_equal(v[i], v[j]));
            }
        }
    }
}

// TEST_CASE_METHOD(Fix, "Xpu8::less", "[Xpu8][010]") {
//     for (size_t i = 0; i < v.size(); i++) {
//         for (size_t j = 0; j < v.size(); j++) {
//             if (i < j) {
//                 CHECK(less(v[i], v[j]));
//             } else {
//                 CHECK(!less(v[i], v[j]));
//             }
//         }
//     }
// }

TEST_CASE_METHOD(Fix, "Xpu8::permuted_unsafe", "[Xpu8][011]") {
    CHECK_THAT(
        permuted_unsafe(
            Xpu8({0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
        Equals(
            Xpu8({2, 3, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})));
    CHECK_THAT(
        permuted_unsafe(
            Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            Xpu8({0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
        Equals(
            Xpu8({3, 2, 1, 5, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 3)));
    CHECK_THAT(
        permuted_unsafe(
            Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 225),
            Xpu8({2, 2, 1, 2, 3, 6, 12, 4, 5, 16, 17, 11, 12, 13, 14, 15})),
        Equals(
            Xpu8({5, 5, 2, 5, 1, 6, 12, 4, 0, 225, 225, 11, 12, 13, 14, 15}, 3)));
    CHECK_THAT(
        permuted_unsafe(
            Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 225),
            Xpu8.rev()),
        Equals(
            Xpu8({225,225,225,225,225,225,225,225,225,225,225,225,225,225,225,225,
                   15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  0,  4,  1,  5,  2,  3})));
}
TEST_CASE_METHOD(Fix, "Xpu8::permuted_ref", "[Xpu8][011]") {
    CHECK_THAT(
        permuted_ref(Xpu8({0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                 Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
        Equals(Xpu8({2, 3, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})));
    CHECK_THAT(
        permuted_ref(Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                 Xpu8({0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
        Equals(Xpu8({3, 2, 1, 5, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 3)));
    CHECK_THAT(
        permuted_ref(Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                          225),
                 Xpu8({2, 2, 1, 2, 3, 6, 12, 4, 5, 16, 17, 11, 12, 13, 14, 15})),
        Equals(Xpu8({5, 5, 2, 5, 1, 6, 12, 4, 0, 225, 225, 11, 12, 13, 14, 15}, 3)));
    CHECK_THAT(
        permuted_ref(Xpu8({3, 2, 5, 1, 4, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 225),
                 Xpu8.rev()),
        Equals(Xpu8({225,225,225,225,225,225,225,225,225,225,225,225,225,225,225,225,
                     15,14,13,12,11,10, 9, 8, 7, 6, 0, 4, 1, 5, 2, 3})));
}
TEST_AGREES2_FUN_EPU8(Fix, permuted, permuted_ref, v, "[Xpu8][012]")

// TEST_CASE_METHOD(Fix, "Xpu8::shifted_left", "[Xpu8][013]") {
//     CHECK_THAT(shifted_left(P01), Equals(P10));
//     CHECK_THAT(shifted_left(P112),
//                Equals(xpu8{1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0}));
//     CHECK_THAT(shifted_left(Pv), Equals(xpu8{5, 2, 5, 1, 6, 12, 4, 0, 3, 2, 11,
//                                              12, 13, 14, 15, 0}));
// }

// TEST_CASE_METHOD(Fix, "Xpu8::shifted_right", "[Xpu8][014]") {
//     CHECK_THAT(shifted_right(P10), Equals(P01));
//     CHECK_THAT(shifted_right(P112), Equals(Xpu8({0, 1, 1}, 2)));
//     CHECK_THAT(shifted_right(Pv), Equals(xpu8{0, 5, 5, 2, 5, 1, 6, 12, 4, 0, 3,
//                                               2, 11, 12, 13, 14}));
// }

// TEST_CASE_METHOD(Fix, "Xpu8::reverted", "[Xpu8][015]") {
//     CHECK_THAT(reverted(Xpu8.id()), Equals(Xpu8.rev()));
//     for (auto x : v) {
//         CHECK_THAT(x, Equals(reverted(reverted(x))));
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::as_array", "[Xpu8][016]") {
//     xpu8 x = Xpu8({4, 2, 5, 1, 2, 7, 7, 3, 4, 2}, 1);
//     auto &refx = as_array(x);
//     refx[2] = 42;
//     CHECK_THAT(x, Equals(Xpu8({4, 2, 42, 1, 2, 7, 7, 3, 4, 2}, 1)));
//     std::fill(refx.begin() + 4, refx.end(), 3);
//     CHECK_THAT(x, Equals(Xpu8({4, 2, 42, 1}, 3)));
//     CHECK(av == as_array(Pv));
// }

// TEST_CASE_METHOD(Fix, "Xpu8(array)", "[Xpu8][017]") {
//     for (auto x : v) {
//         CHECK_THAT(x, Equals(Xpu8(as_array(x))));
//     }
//     CHECK_THAT(Pv, Equals(Xpu8(av)));
// }

// TEST_CASE_METHOD(Fix, "Xpu8::is_sorted", "[Xpu8][018]") {
//     CHECK(is_sorted(Xpu8.id()));
//     CHECK(
//         is_sorted(xpu8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
//     CHECK(is_sorted(Xpu8({0, 1}, 2)));
//     CHECK(is_sorted(Xpu8({0}, 1)));
//     CHECK(is_sorted(Xpu8({}, 5)));
//     CHECK(
//         !is_sorted(xpu8{0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
//     CHECK(!is_sorted(Xpu8({0, 2}, 1)));
//     CHECK(!is_sorted(Xpu8({0, 0, 2}, 1)));
//     CHECK(!is_sorted(Xpu8({6}, 5)));

//     xpu8 x = Xpu8.id();
//     CHECK(is_sorted(x));
//     auto &refx = as_array(x);
// #ifndef __clang__
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wstringop-overflow"
// #endif
//     while (std::next_permutation(refx.begin(), refx.begin() + 9)) {
//         CHECK(!is_sorted(x));
//     }
//     x = Xpu8.id();
//     while (std::next_permutation(refx.begin() + 8, refx.begin() + 16)) {
//         CHECK(!is_sorted(x));
//     }
//     x = sorted(Pa1);
//     CHECK(is_sorted(x));
//     while (std::next_permutation(refx.begin(), refx.begin() + 14)) {
//         CHECK(!is_sorted(x));
//     }
// #ifndef __clang__
// #pragma GCC diagnostic pop
// #endif
// }

// TEST_CASE_METHOD(Fix, "Xpu8::sorted", "[Xpu8][019]") {
//     CHECK_THAT(
//         sorted(xpu8{0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
//         Equals(Xpu8.id()));
//     for (auto &x : v) {
//         CHECK_THAT(sorted(x), IsSorted);
//     }
//     xpu8 x = Xpu8.id();
//     CHECK_THAT(sorted(x), IsSorted);
//     auto &refx = as_array(x);
//     do {
//         CHECK_THAT(sorted(x), IsSorted);
// #ifndef __clang__
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wstringop-overflow"
// #endif
//     } while (std::next_permutation(refx.begin(), refx.begin() + 9));
// #ifndef __clang__
// #pragma GCC diagnostic pop
// #endif
// }

// TEST_CASE_METHOD(Fix, "Xpu8::revsorted", "[Xpu8][020]") {
//     CHECK_THAT(
//         revsorted(xpu8{0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
//         Equals(Xpu8.rev()));
//     for (auto &x : v) {
//         CHECK_THAT(reverted(revsorted(x)), IsSorted);
//     }
//     xpu8 x = Xpu8.id();
//     CHECK_THAT(x, IsSorted);
//     auto &refx = as_array(x);
//     do {
//         CHECK_THAT(reverted(revsorted(x)), IsSorted);
// #ifndef __clang__
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wstringop-overflow"
// #endif
//     } while (std::next_permutation(refx.begin(), refx.begin() + 9));
// #ifndef __clang__
// #pragma GCC diagnostic pop
// #endif
// }

// TEST_CASE_METHOD(Fix, "Xpu8::sort_perm", "[Xpu8][021]") {
//     xpu8 ve{2, 1, 3, 2, 4, 1, 1, 4, 2, 0, 1, 2, 1, 3, 4, 0};
//     CHECK_THAT(sort_perm(ve), Equals(xpu8{9, 15, 1, 5, 6, 10, 12, 3, 0, 8, 11,
//                                           2, 13, 7, 4, 14}));
//     CHECK_THAT(ve,
//                Equals(xpu8{0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4}));

//     for (auto x : v) {
//         xpu8 xsort = x;
//         xpu8 psort = sort_perm(xsort);
//         CHECK_THAT(xsort, IsSorted);
//         CHECK(is_permutation(psort));
//         CHECK_THAT(permuted(x, psort), Equals(xsort));
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::sort8_perm", "[Xpu8][022]") {
//     xpu8 ve{2, 1, 3, 2, 4, 1, 1, 4, 2, 0, 1, 2, 1, 3, 4, 0};
//     CHECK_THAT(sort8_perm(ve), Equals(xpu8{1, 6, 5, 0, 3, 2, 4, 7, 9, 15, 10,
//                                            12, 8, 11, 13, 14}));
//     CHECK_THAT(ve,
//                Equals(xpu8{1, 1, 1, 2, 2, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 4}));

//     for (auto x : v) {
//         xpu8 xsort = x;
//         xpu8 psort = sort_perm(xsort);
//         CHECK_THAT(xsort | Xpu8({0, 0, 0, 0, 0, 0, 0, 0}, 0xFF), IsSorted);
//         CHECK_THAT(xsort & Xpu8({0, 0, 0, 0, 0, 0, 0, 0}, 0xFF), IsSorted);
//         CHECK(is_permutation(psort));
//         CHECK_THAT(permuted(x, psort), Equals(xsort));
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::permutation_of", "[Xpu8][023]") {
//     CHECK_THAT(permutation_of(Xpu8.id(), Xpu8.id()), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of(Pa, Pa), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of(Xpu8.rev(), Xpu8.id()), Equals(Xpu8.rev()));
//     CHECK_THAT(permutation_of(Xpu8.id(), Xpu8.rev()), Equals(Xpu8.rev()));
//     CHECK_THAT(permutation_of(Xpu8.rev(), Xpu8.rev()), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of(Xpu8.id(), RP), Equals(RP));
//     const uint8_t FF = 0xff;
//     CHECK_THAT((permutation_of(Pv, Pv) |
//                 xpu8{FF, FF, FF, FF, 0, 0, FF, 0, 0, 0, FF, 0, FF, 0, 0, 0}),
//                Equals(xpu8{FF, FF, FF, FF, 4, 5, FF, 7, 8, 9, FF, 11, FF, 13,
//                            14, 15}));
// }
// TEST_CASE_METHOD(Fix, "Xpu8::permutation_of_ref", "[Xpu8][024]") {
//     CHECK_THAT(permutation_of_ref(Xpu8.id(), Xpu8.id()), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of_ref(Pa, Pa), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of_ref(Xpu8.rev(), Xpu8.id()), Equals(Xpu8.rev()));
//     CHECK_THAT(permutation_of_ref(Xpu8.id(), Xpu8.rev()), Equals(Xpu8.rev()));
//     CHECK_THAT(permutation_of_ref(Xpu8.rev(), Xpu8.rev()), Equals(Xpu8.id()));
//     CHECK_THAT(permutation_of_ref(Xpu8.id(), RP), Equals(RP));
//     const uint8_t FF = 0xff;
//     CHECK_THAT((permutation_of_ref(Pv, Pv) |
//                 xpu8{FF, FF, FF, FF, 0, 0, FF, 0, 0, 0, FF, 0, FF, 0, 0, 0}),
//                Equals(xpu8{FF, FF, FF, FF, 4, 5, FF, 7, 8, 9, FF, 11, FF, 13,
//                            14, 15}));
// }

// TEST_CASE_METHOD(Fix, "Xpu8::merge", "[Xpu8][025]") {
//     std::vector<std::pair<xpu8, xpu8>> sample_pairs {{
//             { xpu8 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
//               xpu8 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
//             }
//         }};
//     for (auto x : v)
//         for (auto y : v)
//             sample_pairs.emplace_back(x, y);
//     for (auto p : sample_pairs) {
//         xpu8 x = p.first;
//         xpu8 y = p.second;
//         x = sorted(x);
//         y = sorted(y);
//         merge(x, y);
//         CHECK_THAT(x, IsSorted);
//         CHECK_THAT(y, IsSorted);
//         CHECK(x[15] <= y[0]);
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::remove_dups", "[Xpu8][026]") {
//     CHECK_THAT(remove_dups(P1), Equals(P10));
//     CHECK_THAT(remove_dups(P11), Equals(P10));
//     CHECK_THAT(remove_dups(sorted(P10)),
//                Equals(xpu8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}));
//     CHECK_THAT(remove_dups(sorted(Pv)), Equals(xpu8{0, 1, 2, 0, 3, 4, 5, 0, 0,
//                                                     6, 11, 12, 0, 13, 14, 15}));
//     CHECK_THAT(remove_dups(P1, 1), Equals(P1));
//     CHECK_THAT(remove_dups(P11, 1), Equals(Xpu8({1, 1, 0}, 1)));
//     CHECK_THAT(remove_dups(P11, 42), Equals(Xpu8({1, 42, 0}, 42)));
//     CHECK_THAT(remove_dups(sorted(P10), 1), Equals(P1));
//     CHECK_THAT(
//         remove_dups(sorted(Pv), 7),
//         Equals(xpu8{7, 1, 2, 7, 3, 4, 5, 7, 7, 6, 11, 12, 7, 13, 14, 15}));
//     for (auto x : v) {
//         x = sorted(remove_dups(sorted(x)));
//         CHECK_THAT(x, Equals(sorted(remove_dups(x))));
//     }
//     for (auto x : v) {
//         x = sorted(remove_dups(sorted(x), 42));
//         CHECK_THAT(x, Equals(sorted(remove_dups(x, 42))));
//     }
// }

// TEST_CASE_METHOD(Fix, "Xpu8::horiz_sum_ref", "[Xpu8][027]") {
//     CHECK(horiz_sum_ref(zero) == 0);
//     CHECK(horiz_sum_ref(P01) == 1);
//     CHECK(horiz_sum_ref(Xpu8.id()) == 120);
//     CHECK(horiz_sum_ref(P10) == 1);
//     CHECK(horiz_sum_ref(P11) == 2);
//     CHECK(horiz_sum_ref(P1) == 16);
//     CHECK(horiz_sum_ref(P112) == 30);
//     CHECK(horiz_sum_ref(Pa1) == 43);
//     CHECK(horiz_sum_ref(Pa2) == 45);
//     CHECK(horiz_sum_ref(P51) == 90);
//     CHECK(horiz_sum_ref(Pv) == 110);
//     CHECK(horiz_sum_ref(P5) == 80);
//     CHECK(horiz_sum_ref(Xpu8.rev()) == 120);
//     CHECK(horiz_sum_ref(Pc) == 203);
// }

// TEST_AGREES_FUN(Fix, horiz_sum_ref, horiz_sum_gen, v, "[Xpu8][028]")
// TEST_AGREES_FUN(Fix, horiz_sum_ref, horiz_sum4, v, "[Xpu8][029]")
// TEST_AGREES_FUN(Fix, horiz_sum_ref, horiz_sum3, v, "[Xpu8][030]")
// TEST_AGREES_FUN(Fix, horiz_sum_ref, horiz_sum, v, "[Xpu8][031]")

// TEST_CASE_METHOD(Fix, "Xpu8::partial_sums_ref", "[Xpu8][032]") {
//     CHECK_THAT(partial_sums_ref(zero), Equals(zero));
//     CHECK_THAT(partial_sums_ref(P01), Equals(Xpu8({0}, 1)));
//     CHECK_THAT(partial_sums_ref(Xpu8.id()),
//                Equals(xpu8{0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91,
//                            105, 120}));
//     CHECK_THAT(partial_sums_ref(P10), Equals(P1));
//     CHECK_THAT(partial_sums_ref(P11), Equals(Xpu8({1}, 2)));
//     CHECK_THAT(partial_sums_ref(P1), Equals(Xpu8.id() + Xpu8({}, 1)));
//     CHECK_THAT(partial_sums_ref(P112),
//                Equals(xpu8{1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
//                            28, 30}));
//     CHECK_THAT(partial_sums_ref(Pa1),
//                Equals(xpu8{4, 6, 11, 12, 14, 21, 28, 31, 35, 37, 38, 39, 40, 41,
//                            42, 43}));

//     CHECK_THAT(partial_sums_ref(Pa2),
//                Equals(xpu8{4, 6, 11, 12, 14, 23, 30, 33, 37, 39, 40, 41, 42, 43,
//                            44, 45}));
//     CHECK_THAT(partial_sums_ref(P51),
//                Equals(xpu8{5, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78,
//                            84, 90}));
//     CHECK_THAT(partial_sums_ref(Pv),
//                Equals(xpu8{5, 10, 12, 17, 18, 24, 36, 40, 40, 43, 45, 56, 68,
//                            81, 95, 110}));
//     CHECK_THAT(partial_sums_ref(P5),
//                Equals(xpu8{5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
//                            70, 75, 80}));
//     CHECK_THAT(partial_sums_ref(Xpu8.rev()),
//                Equals(xpu8{15, 29, 42, 54, 65, 75, 84, 92, 99, 105, 110, 114,
//                            117, 119, 120, 120}));
//     CHECK_THAT(partial_sums_ref(Pc),
//                Equals(xpu8{23, 28, 49, 54, 97, 133, 140, 147, 154, 161, 168,
//                            175, 182, 189, 196, 203}));
// }
// TEST_AGREES_FUN_XPU8(Fix, partial_sums_ref, partial_sums_gen, v, "[Xpu8][033]")
// TEST_AGREES_FUN_XPU8(Fix, partial_sums_ref, partial_sums_round, v,
//                      "[Xpu8][034]")
// TEST_AGREES_FUN_XPU8(Fix, partial_sums_ref, partial_sums, v, "[Xpu8][035]")

// TEST_CASE_METHOD(Fix, "Xpu8::horiz_max_ref", "[Xpu8][036]") {
//     CHECK(horiz_max_ref(zero) == 0);
//     CHECK(horiz_max_ref(P01) == 1);
//     CHECK(horiz_max_ref(Xpu8.id()) == 15);
//     CHECK(horiz_max_ref(P10) == 1);
//     CHECK(horiz_max_ref(P11) == 1);
//     CHECK(horiz_max_ref(P1) == 1);
//     CHECK(horiz_max_ref(P112) == 2);
//     CHECK(horiz_max_ref(Pa1) == 7);
//     CHECK(horiz_max_ref(Pa2) == 9);
//     CHECK(horiz_max_ref(P51) == 6);
//     CHECK(horiz_max_ref(Pv) == 15);
//     CHECK(horiz_max_ref(P5) == 5);
//     CHECK(horiz_max_ref(Xpu8.rev()) == 15);
//     CHECK(horiz_max_ref(Pc) == 43);
// }

// TEST_AGREES_FUN(Fix, horiz_max_ref, horiz_max_gen, v, "[Xpu8][037]")
// TEST_AGREES_FUN(Fix, horiz_max_ref, horiz_max4, v, "[Xpu8][038]")
// TEST_AGREES_FUN(Fix, horiz_max_ref, horiz_max3, v, "[Xpu8][039]")
// TEST_AGREES_FUN(Fix, horiz_max_ref, horiz_max, v, "[Xpu8][040]")

// TEST_CASE_METHOD(Fix, "Xpu8::partial_max_ref", "[Xpu8][041]") {
//     CHECK_THAT(partial_max_ref(zero), Equals(zero));
//     CHECK_THAT(partial_max_ref(P01), Equals(Xpu8({0}, 1)));
//     CHECK_THAT(partial_max_ref(Xpu8.id()), Equals(Xpu8.id()));
//     CHECK_THAT(partial_max_ref(P10), Equals(P1));
//     CHECK_THAT(partial_max_ref(P11), Equals(P1));
//     CHECK_THAT(partial_max_ref(P1), Equals(P1));
//     CHECK_THAT(partial_max_ref(P112), Equals(P112));
//     CHECK_THAT(partial_max_ref(Pa1), Equals(Xpu8({4, 4, 5, 5, 5}, 7)));
//     CHECK_THAT(partial_max_ref(Pa2), Equals(Xpu8({4, 4, 5, 5, 5}, 9)));
//     CHECK_THAT(partial_max_ref(P51), Equals(Xpu8({5, 5}, 6)));
//     CHECK_THAT(partial_max_ref(Pv), Equals(xpu8{5, 5, 5, 5, 5, 6, 12, 12, 12,
//                                                 12, 12, 12, 12, 13, 14, 15}));
//     CHECK_THAT(partial_max_ref(P5), Equals(P5));
//     CHECK_THAT(partial_max_ref(Xpu8.rev()), Equals(Xpu8({}, 15)));
//     CHECK_THAT(partial_max_ref(Pc), Equals(Xpu8({23, 23, 23, 23}, 43)));
// }
// TEST_AGREES_FUN_XPU8(Fix, partial_max_ref, partial_max_gen, v, "[Xpu8][042]")
// TEST_AGREES_FUN_XPU8(Fix, partial_max_ref, partial_max_round, v, "[Xpu8][043]")
// TEST_AGREES_FUN_XPU8(Fix, partial_max_ref, partial_max, v, "[Xpu8][044]")

// TEST_CASE_METHOD(Fix, "Xpu8::horiz_min_ref", "[Xpu8][045]") {
//     CHECK(horiz_min_ref(zero) == 0);
//     CHECK(horiz_min_ref(P01) == 0);
//     CHECK(horiz_min_ref(Xpu8.id()) == 0);
//     CHECK(horiz_min_ref(P10) == 0);
//     CHECK(horiz_min_ref(P11) == 0);
//     CHECK(horiz_min_ref(P1) == 1);
//     CHECK(horiz_min_ref(P112) == 1);
//     CHECK(horiz_min_ref(Pa1) == 1);
//     CHECK(horiz_min_ref(Pa2) == 1);
//     CHECK(horiz_min_ref(P51) == 1);
//     CHECK(horiz_min_ref(Pv) == 0);
//     CHECK(horiz_min_ref(P5) == 5);
//     CHECK(horiz_min_ref(Xpu8.rev()) == 0);
//     CHECK(horiz_min_ref(Pc) == 5);
// }

// TEST_AGREES_FUN(Fix, horiz_min_ref, horiz_min_gen, v, "[Xpu8][046]")
// TEST_AGREES_FUN(Fix, horiz_min_ref, horiz_min4, v, "[Xpu8][047]")
// TEST_AGREES_FUN(Fix, horiz_min_ref, horiz_min3, v, "[Xpu8][048]")
// TEST_AGREES_FUN(Fix, horiz_min_ref, horiz_min, v, "[Xpu8][049]")

// TEST_CASE_METHOD(Fix, "Xpu8::partial_min_ref", "[Xpu8][050]") {
//     CHECK_THAT(partial_min_ref(zero), Equals(zero));
//     CHECK_THAT(partial_min_ref(P01), Equals(zero));
//     CHECK_THAT(partial_min_ref(Xpu8.id()), Equals(zero));
//     CHECK_THAT(partial_min_ref(P10), Equals(P10));
//     CHECK_THAT(partial_min_ref(P11), Equals(P11));
//     CHECK_THAT(partial_min_ref(P1), Equals(P1));
//     CHECK_THAT(partial_min_ref(P112), Equals(P1));
//     CHECK_THAT(partial_min_ref(Pa1), Equals(Xpu8({4, 2, 2}, 1)));
//     CHECK_THAT(partial_min_ref(Pa2), Equals(Xpu8({4, 2, 2}, 1)));
//     CHECK_THAT(partial_min_ref(P51), Equals(Xpu8({5}, 1)));
//     CHECK_THAT(partial_min_ref(Pv),  // clang-format off
//                  Equals(Xpu8({5, 5, 2, 2, 1, 1, 1, 1, }, 0)));
//     // clang-format on
//     CHECK_THAT(partial_min_ref(P5), Equals(P5));
//     CHECK_THAT(partial_min_ref(Xpu8.rev()), Equals(Xpu8.rev()));
//     CHECK_THAT(partial_min_ref(Pc), Equals(Xpu8({23}, 5)));
// }
// TEST_AGREES_FUN_XPU8(Fix, partial_min_ref, partial_min_gen, v, "[Xpu8][051]")
// TEST_AGREES_FUN_XPU8(Fix, partial_min_ref, partial_min_round, v, "[Xpu8][052]")
// TEST_AGREES_FUN_XPU8(Fix, partial_min_ref, partial_min, v, "[Xpu8][053]")

// TEST_CASE_METHOD(Fix, "Xpu8::eval16_ref", "[Xpu8][054]") {
//     CHECK_THAT(eval16_ref(zero), Equals(Xpu8({16}, 0)));
//     CHECK_THAT(eval16_ref(P01), Equals(Xpu8({15, 1}, 0)));
//     CHECK_THAT(eval16_ref(Xpu8.id()), Equals(Xpu8({}, 1)));
//     CHECK_THAT(eval16_ref(P10), Equals(Xpu8({15, 1}, 0)));
//     CHECK_THAT(eval16_ref(P11), Equals(Xpu8({14, 2}, 0)));
//     CHECK_THAT(eval16_ref(P1), Equals(Xpu8({0, 16}, 0)));
//     CHECK_THAT(eval16_ref(P112), Equals(Xpu8({0, 2, 14}, 0)));
//     CHECK_THAT(eval16_ref(Pa1), Equals(Xpu8({0, 7, 3, 1, 2, 1, 0, 2}, 0)));
//     CHECK_THAT(eval16_ref(Pa2),
//                Equals(Xpu8({0, 7, 3, 1, 2, 1, 0, 1, 0, 1}, 0)));
//     CHECK_THAT(eval16_ref(P51), Equals(Xpu8({0, 1, 0, 0, 0, 1, 14}, 0)));
//     CHECK_THAT(eval16_ref(Pv),
//                Equals(xpu8{1, 1, 2, 1, 1, 3, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1}));
//     CHECK_THAT(eval16_ref(P5), Equals(Xpu8({0, 0, 0, 0, 0, 16}, 0)));
//     CHECK_THAT(eval16_ref(Xpu8.rev()), Equals(Xpu8({}, 1)));
//     CHECK_THAT(eval16_ref(Pc), Equals(Xpu8({0, 0, 0, 0, 0, 2, 0, 10}, 0)));
// }

// TEST_AGREES_FUN_XPU8(Fix, eval16_ref, eval16_cycle, v, "[Xpu8][055]")
// TEST_AGREES_FUN_XPU8(Fix, eval16_ref, eval16_popcount, v, "[Xpu8][056]")
// TEST_AGREES_FUN_XPU8(Fix, eval16_ref, eval16_arr, v, "[Xpu8][057]")
// TEST_AGREES_FUN_XPU8(Fix, eval16_ref, eval16_gen, v, "[Xpu8][058]")
// TEST_AGREES_FUN_XPU8(Fix, eval16_ref, eval16, v, "[Xpu8][059]")

// TEST_CASE("Xpu8::popcount", "[Xpu8][060]") {
//     CHECK_THAT(Xpu8.popcount(),
//                Equals(xpu8{0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4}));
// }

// TEST_CASE_METHOD(Fix, "Xpu8::popcount16", "[Xpu8][061]") {
//     CHECK_THAT(popcount16(Pv),
//                Equals(xpu8{2, 2, 1, 2, 1, 2, 2, 1, 0, 2, 1, 3, 2, 3, 3, 4}));
//     CHECK_THAT(popcount16(RP),
//                Equals(xpu8{2, 1, 0, 3, 4, 3, 2, 2, 1, 3, 2, 2, 3, 1, 1, 2}));
//     CHECK_THAT(popcount16(RP << 1),
//                Equals(xpu8{2, 1, 0, 3, 4, 3, 2, 2, 1, 3, 2, 2, 3, 1, 1, 2}));
//     CHECK_THAT(popcount16(RP << 2),
//                Equals(xpu8{2, 1, 0, 3, 4, 3, 2, 2, 1, 3, 2, 2, 3, 1, 1, 2}));
//     CHECK_THAT(popcount16(Xpu8({0, 1, 5, 0xff, 0xf0, 0x35}, 0x0f)),
//                Equals(Xpu8({0, 1, 2, 8}, 4)));
// }

// TEST_CASE("random_xpu8", "[Xpu8][062]") {
//     for (int bnd : {1, 10, 100, 255, 256}) {
//         for (int i = 0; i < 10; i++) {
//             xpu8 r = random_xpu8(bnd);
//             CHECK_THAT(r, Equals(r));
//             for (auto v : as_array(r))
//                 CHECK(v < bnd);
//         }
//     }
// }

// TEST_CASE_METHOD(Fix, "is_partial_transformation", "[Xpu8][063]") {
//     CHECK(is_partial_transformation(zero));
//     CHECK(is_partial_transformation(P01));
//     CHECK(is_partial_transformation(P10));
//     CHECK(!is_partial_transformation(Xpu8({16}, 0)));
//     CHECK(is_partial_transformation(Xpu8({}, 0xff)));
//     CHECK(is_partial_transformation(Xpu8({2, 0xff, 3}, 0)));

//     CHECK(!is_partial_transformation(zero, 15));
//     CHECK(is_partial_transformation(Pa));
//     CHECK(is_partial_transformation(Pa, 6));
//     CHECK(is_partial_transformation(Pa, 5));
//     CHECK(!is_partial_transformation(Pa, 4));
//     CHECK(!is_partial_transformation(Pa, 1));
//     CHECK(!is_partial_transformation(Pa, 0));

//     CHECK(is_partial_transformation(RP));
//     CHECK(is_partial_transformation(RP, 16));
//     CHECK(!is_partial_transformation(RP, 15));
//     CHECK(is_partial_transformation(Xpu8({1, 2, 1, 0xFF, 0, 5, 0xFF, 2}, 0)));
//     CHECK(!is_partial_transformation(Xpu8({1, 2, 1, 0xFF, 0, 16, 0xFF, 2}, 0)));
// }

// TEST_CASE_METHOD(Fix, "is_transformation", "[Xpu8][064]") {
//     CHECK(is_transformation(zero));
//     CHECK(is_transformation(P01));
//     CHECK(is_transformation(P10));
//     CHECK(!is_transformation(Xpu8({16}, 0)));
//     CHECK(!is_transformation(Xpu8({}, 0xff)));
//     CHECK(!is_transformation(Xpu8({2, 0xff, 3}, 0)));

//     CHECK(!is_transformation(zero, 15));
//     CHECK(is_transformation(Pa));
//     CHECK(is_transformation(Pa, 6));
//     CHECK(is_transformation(Pa, 5));
//     CHECK(!is_transformation(Pa, 4));
//     CHECK(!is_transformation(Pa, 1));
//     CHECK(!is_transformation(Pa, 0));

//     CHECK(is_transformation(RP));
//     CHECK(is_transformation(RP, 16));
//     CHECK(!is_transformation(RP, 15));
// }

// TEST_CASE_METHOD(Fix, "is_partial_permutation", "[Xpu8][065]") {
//     CHECK(!is_partial_permutation(zero));
//     CHECK(!is_partial_permutation(P01));
//     CHECK(!is_partial_permutation(P10));
//     CHECK(!is_partial_permutation(Xpu8({16}, 0)));
//     CHECK(is_partial_permutation(Xpu8({}, 0xff)));
//     CHECK(!is_partial_permutation(Xpu8({2, 0xff, 3}, 0)));
//     CHECK(is_partial_permutation(Xpu8({2, 0xff, 3}, 0xff)));

//     CHECK(!is_partial_permutation(zero, 15));
//     CHECK(is_partial_permutation(Pa));
//     CHECK(is_partial_permutation(Pa, 6));
//     CHECK(is_partial_permutation(Pa, 5));
//     CHECK(!is_partial_permutation(Pa, 4));
//     CHECK(!is_partial_permutation(Pa, 1));
//     CHECK(!is_partial_permutation(Pa, 0));

//     CHECK(is_partial_permutation(RP));
//     CHECK(is_partial_permutation(RP, 16));
//     CHECK(!is_partial_permutation(RP, 15));

//     CHECK(is_partial_permutation(
//         xpu8{1, 2, 0xFF, 0xFF, 0, 5, 0xFF, 3, 8, 9, 10, 11, 12, 13, 14, 15}));
//     CHECK(!is_partial_permutation(
//         xpu8{1, 2, 1, 0xFF, 0, 5, 0xFF, 2, 8, 9, 10, 11, 12, 13, 14, 15}));
//     CHECK(!is_partial_permutation(Xpu8({1, 2, 1, 0xFF, 0, 5, 0xFF, 2}, 0)));
//     CHECK(!is_partial_permutation(Xpu8({1, 2, 1, 0xFF, 0, 16, 0xFF, 2}, 0)));
// }

// TEST_CASE_METHOD(Fix, "is_permutation", "[Xpu8][066]") {
//     CHECK(!is_permutation(zero));
//     CHECK(!is_permutation(P01));
//     CHECK(!is_permutation(P10));
//     CHECK(!is_permutation(Xpu8({16}, 0)));
//     CHECK(!is_permutation(Xpu8({}, 0xff)));
//     CHECK(!is_permutation(Xpu8({2, 0xff, 3}, 0)));

//     CHECK(!is_permutation(zero, 15));
//     CHECK(is_permutation(Pa));
//     CHECK(is_permutation(Pa, 6));
//     CHECK(is_permutation(Pa, 5));
//     CHECK(!is_permutation(Pa, 4));
//     CHECK(!is_permutation(Pa, 1));
//     CHECK(!is_permutation(Pa, 0));

//     CHECK(is_permutation(RP));
//     CHECK(is_permutation(RP, 16));
//     CHECK(!is_permutation(RP, 15));
// }

// #ifdef SIMDE_X86_SSE4_2_NATIVE
// TEST_CASE_METHOD(Fix, "is_permutation_cmpestri", "[Xpu8][067]") {
//     for (auto x : v) {
//         for (size_t i = 0; i < 16; i++) {
//             CHECK(is_permutation_cmpestri(x, i) == is_permutation(x, i));
//         }
//     }
// }
// #endif

// TEST_CASE_METHOD(Fix, "is_permutation_sort", "[Xpu8][068]") {
//     for (auto x : v) {
//         for (size_t i = 0; i < 16; i++) {
//             CHECK(is_permutation_sort(x, i) == is_permutation(x, i));
//         }
//     }
// }

// TEST_CASE_METHOD(Fix, "is_permutation_eval", "[Xpu8][069]") {
//     for (auto x : v) {
//         for (size_t i = 0; i < 16; i++) {
//             CHECK(is_permutation_eval(x, i) == is_permutation(x, i));
//         }
//     }
// }

}  // namespace HPCombi
