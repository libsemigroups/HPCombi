#include <benchmark/benchmark.h>
//~ #include "perm16.hpp"
//~ #include "perm_generic.hpp"
#include "bench_fixture.hpp"
#include "iostream"
#include "vect_generic.hpp"

#include <string.h>
#include <stdlib.h>

using namespace std;
using HPCombi::epu8;

const Fix_perm16 bench_data;

#define ASSERT(test) if (!(test)) cout << "Test failed in file " << __FILE__ \
                                       << " line " << __LINE__ << ": " #test << endl


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
    static const epu8 even =
        {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    static const epu8 odd =
        {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15};
    static const epu8 mask =
        {0, FF, 0, FF, 0, FF, 0, FF, 0, FF, 0, FF, 0, FF, 0, FF};
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
    std::array<uint8_t, 16> stat {};
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
    HPCombi::VectGeneric<16> &ar =
        reinterpret_cast<HPCombi::VectGeneric<16>&>(HPCombi::as_array(p));
    ar.sort();
    return p;
}


template<typename TF, typename Sample>
void myBench(const char* name, TF pfunc, const char* label, Sample &sample) {
    benchmark::RegisterBenchmark(name,
        [pfunc](benchmark::State& st, const char* label, Sample &sample) {
            for (auto _ : st) {
                for (auto elem : sample) {
                    benchmark::DoNotOptimize(pfunc(elem));
                }
            }
            st.SetLabel(label);
        }, label, sample);
}

static const epu8 bla = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15};

#define MYBENCH(nm, fun, lbl, smp)  \
    myBench(nm, [](epu8 p) { return fun(p); }, lbl, smp)
#define MYBENCH2(nm, fun, lbl, smp) \
    myBench(nm, [](epu8 p) { return fun(p,bla); }, lbl, smp)

//##################################################################################
int Bench_sort() {
    myBench("sort_ref", std_sort, "std1", bench_data.sample);
    myBench("sort_ref", std_sort, "std2", bench_data.sample);
    myBench("sort_ref", std_sort, "std3", bench_data.sample);

    myBench("sort_alt", std_sort, "std", bench_data.sample);
    myBench("sort_alt", arr_sort, "arr", bench_data.sample);
    myBench("sort_alt", gen_sort, "gen", bench_data.sample);
    myBench("sort_alt", insertion_sort, "insert", bench_data.sample);
    myBench("sort_alt", sort_odd_even, "odd_even", bench_data.sample);
    myBench("sort_alt", radix_sort, "radix", bench_data.sample);
    myBench("sort_alt", sort_pair, "pair", bench_data.sample);
    myBench("sort_alt", HPCombi::sorted, "netw", bench_data.sample);

    // lambda function is needed for inlining
    MYBENCH("sort_lmbd", std_sort, "std", bench_data.sample);
    MYBENCH("sort_lmbd", arr_sort, "arr", bench_data.sample);
    MYBENCH("sort_lmbd", gen_sort, "gen", bench_data.sample);
    MYBENCH("sort_lmbd", insertion_sort, "insert", bench_data.sample);
    MYBENCH("sort_lmbd", sort_odd_even, "odd_even", bench_data.sample);
    MYBENCH("sort_lmbd", radix_sort, "radix", bench_data.sample);
    MYBENCH("sort_lmbd", sort_pair, "pair", bench_data.sample);
    MYBENCH("sort_lmbd", HPCombi::sorted, "netw", bench_data.sample);
    return 0;
}

//##################################################################################
int Bench_hsum() {
    myBench("hsum_ref", HPCombi::horiz_sum_ref, "ref1", bench_data.sample);
    myBench("hsum_ref", HPCombi::horiz_sum_ref, "ref2", bench_data.sample);
    myBench("hsum_ref", HPCombi::horiz_sum_ref, "ref3", bench_data.sample);

    myBench("hsum_alt", HPCombi::horiz_sum_ref, "ref", bench_data.sample);
    myBench("hsum_alt", HPCombi::horiz_sum_arr, "arr", bench_data.sample);
    myBench("hsum_alt", HPCombi::horiz_sum4, "sum4", bench_data.sample);
    myBench("hsum_alt", HPCombi::horiz_sum3, "sum3", bench_data.sample);

    MYBENCH("hsum_lmbd", HPCombi::horiz_sum_ref, "ref", bench_data.sample);
    MYBENCH("hsum_lmbd", HPCombi::horiz_sum_arr, "arr", bench_data.sample);
    MYBENCH("hsum_lmbd", HPCombi::horiz_sum4, "sum4", bench_data.sample);
    MYBENCH("hsum_lmbd", HPCombi::horiz_sum3, "sum3", bench_data.sample);
    return 0;
}


//##################################################################################
int Bench_psum() {
    myBench("psum_ref", HPCombi::partial_sums_ref, "ref1", bench_data.sample);
    myBench("psum_ref", HPCombi::partial_sums_ref, "ref2", bench_data.sample);
    myBench("psum_ref", HPCombi::partial_sums_ref, "ref3", bench_data.sample);

    myBench("psum_alt", HPCombi::partial_sums_ref, "ref", bench_data.sample);
    myBench("psum_alt", HPCombi::partial_sums_round, "rnd", bench_data.sample);

    MYBENCH("psum_lmbd", HPCombi::partial_sums_ref, "ref", bench_data.sample);
    MYBENCH("psum_lmbd", HPCombi::partial_sums_round, "rnd", bench_data.sample);
    return 0;
}

//##################################################################################
int Bench_eval() {
    myBench("eval_ref", HPCombi::eval16_ref, "ref1", bench_data.sample);
    myBench("eval_ref", HPCombi::eval16_ref, "ref2", bench_data.sample);
    myBench("eval_ref", HPCombi::eval16_ref, "ref3", bench_data.sample);

    myBench("eval_alt", HPCombi::eval16_ref, "ref", bench_data.sample);
    myBench("eval_alt", HPCombi::eval16_popcount, "popcnt", bench_data.sample);
    myBench("eval_alt", HPCombi::eval16_arr, "arr", bench_data.sample);
    myBench("eval_alt", HPCombi::eval16_cycle, "cycle", bench_data.sample);

    MYBENCH("eval_lmbd", HPCombi::eval16_ref, "ref", bench_data.sample);
    MYBENCH("eval_lmbd", HPCombi::eval16_popcount, "popcnt", bench_data.sample);
    MYBENCH("eval_lmbd", HPCombi::eval16_arr, "arr", bench_data.sample);
    MYBENCH("eval_lmbd", HPCombi::eval16_cycle, "cycle", bench_data.sample);
    return 0;
}

//##################################################################################
int Bench_first_diff() {
    MYBENCH2("first_diff", HPCombi::first_diff_ref, "ref", bench_data.sample);
    MYBENCH2("first_diff", HPCombi::first_diff_cmpstr, "cmpstr", bench_data.sample);
    MYBENCH2("first_diff", HPCombi::first_diff_mask, "mask", bench_data.sample);
    return 0;
}

//##################################################################################
int Bench_last_diff() {
    MYBENCH2("last_diff", HPCombi::last_diff_ref, "ref", bench_data.sample);
    MYBENCH2("last_diff", HPCombi::last_diff_cmpstr, "cmpstr", bench_data.sample);
    MYBENCH2("last_diff", HPCombi::last_diff_mask, "mask", bench_data.sample);
    return 0;
}

auto dummy = {
    Bench_sort(),
    Bench_hsum(),
    Bench_psum(),
    Bench_eval(),
    Bench_first_diff(),
    Bench_last_diff()
};

BENCHMARK_MAIN();
