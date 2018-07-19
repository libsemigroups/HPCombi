#include <benchmark/benchmark.h>
//~ #include "perm16.hpp"
//~ #include "perm_generic.hpp"
#include "bench_fixture.hpp"
#include "iostream"

#include <string.h>
#include <stdlib.h>

using namespace std;
using HPCombi::epu8;

const Fix_perm16 bench_data;

#define ASSERT(test) if (!(test)) cout << "Test failed in file " << __FILE__ \
                                       << " line " << __LINE__ << ": " #test << endl

template<typename TF, typename Sample>
void bench_sort(const char* name, TF pfunc, const char* label, Sample &sample) {
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
    auto &a = HPCombi::epu8cons.as_array(p);
    for (int i = 0; i < 16; i++)
        for (int j = i; j > 0 && a[j] < a[j - 1]; j--)
            std::swap(a[j], a[j - 1]);
    return p;
}

inline epu8 radix_sort(epu8 p) {
    auto &a = HPCombi::epu8cons.as_array(p);
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
    auto &ar = HPCombi::epu8cons.as_array(p);
    std::sort(ar.begin(), ar.end());
    return p;
}

//##################################################################################
int Bench_sort() {
    bench_sort("sort_ref", std_sort, "std", bench_data.sample);

    bench_sort("sort_alt", std_sort, "std", bench_data.sample);
    bench_sort("sort_alt", insertion_sort, "insert", bench_data.sample);
    bench_sort("sort_alt", sort_odd_even, "odd_even", bench_data.sample);
    bench_sort("sort_alt", radix_sort, "radix", bench_data.sample);
    bench_sort("sort_alt", sort_pair, "pair", bench_data.sample);
    bench_sort("sort_alt", HPCombi::sorted, "netw", bench_data.sample);

    // lambda function is needed for inlining
    bench_sort("sort_lambda",
              [](epu8 p) {return std_sort(p);},
              "std", bench_data.sample);
    bench_sort("sort_lambda",
              [](epu8 p) {return insertion_sort(p);},
              "insert", bench_data.sample);
    bench_sort("sort_lambda",
              [](epu8 p) {return sort_odd_even(p);},
              "odd_even", bench_data.sample);
    bench_sort("sort_lambda",
              [](epu8 p) {return radix_sort(p);},
              "radix", bench_data.sample);
    bench_sort("sort_lambda",
              [](epu8 p) {return sort_pair(p);},
              "pair", bench_data.sample);
    bench_sort("sort_lambda",
              [](epu8 p) {return HPCombi::sorted(p);},
              "netw", bench_data.sample);
    return 0;
}


int dummy1 = Bench_sort();

BENCHMARK_MAIN();
