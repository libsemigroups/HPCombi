//****************************************************************************//
//       Copyright (C) 2018 Florent Hivert <Florent.Hivert@lri.fr>,           //
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

#include <benchmark/benchmark.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "bench_fixture.hpp"
#include "compilerinfo.hpp"
#include "cpu_x86_impl.hpp"

using namespace FeatureDetector;
using namespace std;
using HPCombi::epu8;

const Fix_epu8 sample;
const std::string SIMDSET = cpu_x86::get_highest_SIMD();
const std::string PROCID = cpu_x86::get_proc_string();

struct RoundsMask {
    // commented out due to a bug in gcc
    /* constexpr */ RoundsMask() : arr() {
        for (unsigned i = 0; i < HPCombi::sorting_rounds.size(); ++i)
            arr[i] = HPCombi::sorting_rounds[i] < HPCombi::epu8id;
    }
    epu8 arr[HPCombi::sorting_rounds.size()];
};

const auto rounds_mask = RoundsMask();

inline epu8 sort_pair(epu8 a) {
    for (unsigned i = 0; i < HPCombi::sorting_rounds.size(); ++i) {
        epu8 minab, maxab, b = HPCombi::permuted(a, HPCombi::sorting_rounds[i]);
        minab = _mm_min_epi8(a, b);
        maxab = _mm_max_epi8(a, b);
        a = _mm_blendv_epi8(minab, maxab, rounds_mask.arr[i]);
    }
    return a;
}

inline epu8 sort_odd_even(epu8 a) {
    const uint8_t FF = 0xff;
    static const epu8 even = {1, 0, 3,  2,  5,  4,  7,  6,
                              9, 8, 11, 10, 13, 12, 15, 14};
    static const epu8 odd = {0, 2,  1, 4,  3,  6,  5,  8,
                             7, 10, 9, 12, 11, 14, 13, 15};
    static const epu8 mask = {0, FF, 0, FF, 0, FF, 0, FF,
                              0, FF, 0, FF, 0, FF, 0, FF};
    epu8 b, minab, maxab;
    for (unsigned i = 0; i < 8; ++i) {
        b = HPCombi::permuted(a, even);
        minab = _mm_min_epi8(a, b);
        maxab = _mm_max_epi8(a, b);
        a = _mm_blendv_epi8(minab, maxab, mask);
        b = HPCombi::permuted(a, odd);
        minab = _mm_min_epi8(a, b);
        maxab = _mm_max_epi8(a, b);
        a = _mm_blendv_epi8(maxab, minab, mask);
    }
    return a;
}

inline epu8 insertion_sort(epu8 p) {
    auto &a = HPCombi::as_array(p);
    for (int i = 0; i < 16; i++)
        for (int j = i; j > 0 && a[j] < a[j - 1]; j--)
            std::swap(a[j], a[j - 1]);
    return p;
}

inline epu8 radix_sort(epu8 p) {
    auto &a = HPCombi::as_array(p);
    std::array<uint8_t, 16> stat{};
    for (int i = 0; i < 16; i++)
        stat[a[i]]++;
    int c = 0;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < stat[i]; j++)
            a[c++] = i;
    return p;
}

inline epu8 std_sort(epu8 &p) {
    auto &ar = HPCombi::as_array(p);
    std::sort(ar.begin(), ar.end());
    return p;
}

inline epu8 arr_sort(epu8 &p) {
    auto &ar = HPCombi::as_array(p);
    return HPCombi::from_array(HPCombi::sorted_vect(ar));
}

inline epu8 gen_sort(epu8 p) {
    HPCombi::as_VectGeneric(p).sort();
    return p;
}

template <typename TF, typename Sample>
void myBench(const string &name, TF pfunc, Sample &sample) {
    string fullname = name + "_" + CXX_VER + "_proc-" + PROCID;
    benchmark::RegisterBenchmark(
        fullname.c_str(), [pfunc, sample](benchmark::State &st) {
            for (auto _ : st) {
                for (auto elem : sample) {
                    benchmark::DoNotOptimize(pfunc(elem));
                }
            }
        });
}

static const epu8 bla = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15};

#define MYBENCH(nm, fun, smp)                                                  \
    myBench(                                                                   \
        nm, [](epu8 p) { return fun(p); }, smp)
#define MYBENCH2(nm, fun, smp)                                                 \
    myBench(                                                                   \
        nm, [](epu8 p) { return fun(p, bla); }, smp)

// ##################################################################################
int Bench_sort() {
    myBench("sort_std1_nolmbd", std_sort, sample.perms);
    myBench("sort_std2_nolmbd", std_sort, sample.perms);
    myBench("sort_std3_nolmbd", std_sort, sample.perms);

    myBench("sort_std_nolmbd", std_sort, sample.perms);
    myBench("sort_arr_nolmbd", arr_sort, sample.perms);
    myBench("sort_gen_nolmbd", gen_sort, sample.perms);
    myBench("sort_insert_nolmbd", insertion_sort, sample.perms);
    myBench("sort_oddEven_nolmbd", sort_odd_even, sample.perms);
    myBench("sort_radix_nolmbd", radix_sort, sample.perms);
    myBench("sort_pair_nolmbd", sort_pair, sample.perms);
    myBench("sort_netw_nolmbd", HPCombi::sorted, sample.perms);

    // lambda function is needed for inlining
    MYBENCH("sort_std_lmbd", std_sort, sample.perms);
    MYBENCH("sort_arr_lmbd", arr_sort, sample.perms);
    MYBENCH("sort_gen_lmbd", gen_sort, sample.perms);
    MYBENCH("sort_insert_lmbd", insertion_sort, sample.perms);
    MYBENCH("sort_oddEven_lmbd", sort_odd_even, sample.perms);
    MYBENCH("sort_radix_lmbd", radix_sort, sample.perms);
    MYBENCH("sort_pair_lmbd", sort_pair, sample.perms);
    MYBENCH("sort_netw_lmbd", HPCombi::sorted, sample.perms);

    MYBENCH("sort8_std_lmbd", std_sort, sample.vects);
    MYBENCH("sort8_arr_lmbd", arr_sort, sample.vects);
    MYBENCH("sort8_gen_lmbd", gen_sort, sample.vects);
    MYBENCH("sort8_insert_lmbd", insertion_sort, sample.vects);
    MYBENCH("sort8_oddEven_lmbd", sort_odd_even, sample.vects);
    MYBENCH("sort8_pair_lmbd", sort_pair, sample.vects);
    MYBENCH("sort8_netw_lmbd", HPCombi::sorted, sample.vects);
    return 0;
}

// ##################################################################################
int Bench_hsum() {
    myBench("hsum_ref1_nolmbd", HPCombi::horiz_sum_ref, sample.perms);
    myBench("hsum_ref2_nolmbd", HPCombi::horiz_sum_ref, sample.perms);
    myBench("hsum_ref3_nolmbd", HPCombi::horiz_sum_ref, sample.perms);

    myBench("hsum_ref_nolmbd", HPCombi::horiz_sum_ref, sample.perms);
    myBench("hsum_gen_nolmbd", HPCombi::horiz_sum_gen, sample.perms);
    myBench("hsum_sum4_nolmbd", HPCombi::horiz_sum4, sample.perms);
    myBench("hsum_sum3_nolmbd", HPCombi::horiz_sum3, sample.perms);

    MYBENCH("hsum_ref_lmbd", HPCombi::horiz_sum_ref, sample.perms);
    MYBENCH("hsum_gen_lmbd", HPCombi::horiz_sum_gen, sample.perms);
    MYBENCH("hsum_sum4_lmbd", HPCombi::horiz_sum4, sample.perms);
    MYBENCH("hsum_sum3_lmbd", HPCombi::horiz_sum3, sample.perms);
    return 0;
}
// ##################################################################################
int Bench_psum() {
    myBench("psum_ref1_nolmbd", HPCombi::partial_sums_ref, sample.perms);
    myBench("psum_ref2_nolmbd", HPCombi::partial_sums_ref, sample.perms);
    myBench("psum_ref3_nolmbd", HPCombi::partial_sums_ref, sample.perms);

    myBench("psum_ref_nolmbd", HPCombi::partial_sums_ref, sample.perms);
    myBench("psum_gen_nolmbd", HPCombi::partial_sums_gen, sample.perms);
    myBench("psum_rnd_nolmbd", HPCombi::partial_sums_round, sample.perms);

    MYBENCH("psum_ref_lmbd", HPCombi::partial_sums_ref, sample.perms);
    MYBENCH("psum_gen_lmbd", HPCombi::partial_sums_gen, sample.perms);
    MYBENCH("psum_rnd_lmbd", HPCombi::partial_sums_round, sample.perms);
    return 0;
}

// ##################################################################################
int Bench_hmax() {
    myBench("hmax_ref1_nolmbd", HPCombi::horiz_max_ref, sample.perms);
    myBench("hmax_ref2_nolmbd", HPCombi::horiz_max_ref, sample.perms);
    myBench("hmax_ref3_nolmbd", HPCombi::horiz_max_ref, sample.perms);

    myBench("hmax_ref_nolmbd", HPCombi::horiz_max_ref, sample.perms);
    //    myBench("hmax_gen_nolmbd", HPCombi::horiz_max_gen, sample.perms);
    myBench("hmax_max4_nolmbd", HPCombi::horiz_max4, sample.perms);
    myBench("hmax_max3_nolmbd", HPCombi::horiz_max3, sample.perms);

    MYBENCH("hmax_ref_lmbd", HPCombi::horiz_max_ref, sample.perms);
    //    MYBENCH("hmax_gen_lmbd", HPCombi::horiz_max_gen, sample.perms);
    MYBENCH("hmax_max4_lmbd", HPCombi::horiz_max4, sample.perms);
    MYBENCH("hmax_max3_lmbd", HPCombi::horiz_max3, sample.perms);
    return 0;
}
// ##################################################################################
int Bench_pmax() {
    myBench("pmax_ref1_nolmbd", HPCombi::partial_max_ref, sample.perms);
    myBench("pmax_ref2_nolmbd", HPCombi::partial_max_ref, sample.perms);
    myBench("pmax_ref3_nolmbd", HPCombi::partial_max_ref, sample.perms);

    myBench("pmax_ref_nolmbd", HPCombi::partial_max_ref, sample.perms);
    //    myBench("pmax_gen_nolmbd", HPCombi::partial_max_gen, sample.perms);
    myBench("pmax_rnd_nolmbd", HPCombi::partial_max_round, sample.perms);

    MYBENCH("pmax_ref_lmbd", HPCombi::partial_max_ref, sample.perms);
    //    MYBENCH("pmax_gen_lmbd", HPCombi::partial_max_gen, sample.perms);
    MYBENCH("pmax_rnd_lmbd", HPCombi::partial_max_round, sample.perms);
    return 0;
}

// ##################################################################################
int Bench_hmin() {
    myBench("hmin_ref1_nolmbd", HPCombi::horiz_min_ref, sample.perms);
    myBench("hmin_ref2_nolmbd", HPCombi::horiz_min_ref, sample.perms);
    myBench("hmin_ref3_nolmbd", HPCombi::horiz_min_ref, sample.perms);

    myBench("hmin_ref_nolmbd", HPCombi::horiz_min_ref, sample.perms);
    //    myBench("hmin_gen_nolmbd", HPCombi::horiz_min_gen, sample.perms);
    myBench("hmin_min4_nolmbd", HPCombi::horiz_min4, sample.perms);
    myBench("hmin_min3_nolmbd", HPCombi::horiz_min3, sample.perms);

    MYBENCH("hmin_ref_lmbd", HPCombi::horiz_min_ref, sample.perms);
    //    MYBENCH("hmin_gen_lmbd", HPCombi::horiz_min_gen, sample.perms);
    MYBENCH("hmin_min4_lmbd", HPCombi::horiz_min4, sample.perms);
    MYBENCH("hmin_min3_lmbd", HPCombi::horiz_min3, sample.perms);
    return 0;
}
// ##################################################################################
int Bench_pmin() {
    myBench("pmin_ref1_nolmbd", HPCombi::partial_min_ref, sample.perms);
    myBench("pmin_ref2_nolmbd", HPCombi::partial_min_ref, sample.perms);
    myBench("pmin_ref3_nolmbd", HPCombi::partial_min_ref, sample.perms);

    myBench("pmin_ref_nolmbd", HPCombi::partial_min_ref, sample.perms);
    //    myBench("pmin_gen_nolmbd", HPCombi::partial_min_gen, sample.perms);
    myBench("pmin_rnd_nolmbd", HPCombi::partial_min_round, sample.perms);

    MYBENCH("pmin_ref_lmbd", HPCombi::partial_min_ref, sample.perms);
    //    MYBENCH("pmin_gen_lmbd", HPCombi::partial_min_gen, sample.perms);
    MYBENCH("pmin_rnd_lmbd", HPCombi::partial_min_round, sample.perms);
    return 0;
}

// ##################################################################################
int Bench_eval() {
    myBench("eval_ref1_nolmbd", HPCombi::eval16_ref, sample.perms);
    myBench("eval_ref2_nolmbd", HPCombi::eval16_ref, sample.perms);
    myBench("eval_ref3_nolmbd", HPCombi::eval16_ref, sample.perms);

    myBench("eval_ref_nolmbd", HPCombi::eval16_ref, sample.perms);
    myBench("eval_gen_nolmbd", HPCombi::eval16_gen, sample.perms);
    myBench("eval_popcnt_nolmbd", HPCombi::eval16_popcount, sample.perms);
    myBench("eval_arr_nolmbd", HPCombi::eval16_arr, sample.perms);
    myBench("eval_cycle_nolmbd", HPCombi::eval16_cycle, sample.perms);

    MYBENCH("eval_ref_lmbd", HPCombi::eval16_ref, sample.perms);
    MYBENCH("eval_gen_lmbd", HPCombi::eval16_gen, sample.perms);
    MYBENCH("eval_popcnt_lmbd", HPCombi::eval16_popcount, sample.perms);
    MYBENCH("eval_arr_lmbd", HPCombi::eval16_arr, sample.perms);
    MYBENCH("eval_cycle_lmbd", HPCombi::eval16_cycle, sample.perms);
    return 0;
}

// ##################################################################################
int Bench_first_diff() {
    MYBENCH2("firstDiff_ref_lmbd", HPCombi::first_diff_ref, sample.perms);
    MYBENCH2("firstDiff_cmpstr_lmbd", HPCombi::first_diff_cmpstr, sample.perms);
    MYBENCH2("firstDiff_mask_lmbd", HPCombi::first_diff_mask, sample.perms);
    return 0;
}

// ##################################################################################
int Bench_last_diff() {
    MYBENCH2("lastDiff_ref_lmbd", HPCombi::last_diff_ref, sample.perms);
    MYBENCH2("lastDiff_cmpstr_lmbd", HPCombi::last_diff_cmpstr, sample.perms);
    MYBENCH2("lastDiff_mask_lmbd", HPCombi::last_diff_mask, sample.perms);
    return 0;
}

auto dummy = {Bench_sort(),       Bench_hsum(),     Bench_psum(), Bench_hmax(),
              Bench_pmax(),       Bench_hmin(),     Bench_pmin(), Bench_eval(),
              Bench_first_diff(), Bench_last_diff()};

BENCHMARK_MAIN();
