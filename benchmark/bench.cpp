#include <benchmark/benchmark.h>
//~ #include "perm16.hpp"
//~ #include "perm_generic.hpp"
#include "bench_fixture.hpp"
#include "iostream"

#include <string.h>
#include <stdlib.h>

using namespace std;
using HPCombi::epu8;

const Fix_perm16 perm16_bench_data;

#define ASSERT(test) if (!(test)) cout << "Test failed in file " << __FILE__ \
                                       << " line " << __LINE__ << ": " #test << endl
//##################################################################################
// Register fuction for generic operation that take zeros argument
template<typename TF, typename Sample>
void bench_inv(const char* name, TF pfunc, const char* label, Sample &sample) {
    benchmark::RegisterBenchmark(name,
        [pfunc](benchmark::State& st, const char* label, Sample &sample) {
            for (auto _ : st) {
                bool ok = true;
                for (auto elem : sample) {
                    ok &= HPCombi::equal(pfunc(elem), HPCombi::epu8id);
                }
                //ASSERT(ok);
                benchmark::DoNotOptimize(ok);
            }
            st.SetLabel(label);
            // st.SetItemsProcessed(st.iterations()*perm16_bench_data.sample.size());
        }, label, sample);
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

inline epu8 insertion_sort(epu8 a) {
    for (int i = 0; i < 16; i++)
        for (int j = i; j > 0 && a[j] < a[j - 1]; j--)
            std::swap(a[j], a[j - 1]);
    return a;
}

inline epu8 radix_sort(epu8 a) {
    epu8 stat = {}, res;
    for (int i = 0; i < 16; i++)
        stat[a[i]]++;
    int c = 0;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < stat[i]; j++)
            res[c++] = i;
    return res;
}

//##################################################################################
int Bench_sort() {
    bench_inv("sort_ref",
              [](epu8 p) {
                  auto &ar = HPCombi::epu8cons.as_array(p);
                  std::sort(ar.begin(), ar.end());
                  return p;
              }, "std", perm16_bench_data.sample);
    bench_inv("sort_alt",
              [](epu8 p) {
                  auto &ar = HPCombi::epu8cons.as_array(p);
                  std::sort(ar.begin(), ar.end());
                  return p;
              }, "std", perm16_bench_data.sample);
    bench_inv("sort_alt",
              [](epu8 p) {return insertion_sort(p);},
              "insert_lambda", perm16_bench_data.sample);
    bench_inv("sort_alt",
              [](epu8 p) {return sort_odd_even(p);},
              "odd_even", perm16_bench_data.sample);
    bench_inv("sort_alt",
              [](epu8 p) {return radix_sort(p);},
              "radix", perm16_bench_data.sample);
    bench_inv("sort_alt",
              HPCombi::sorted, "netw", perm16_bench_data.sample);
    bench_inv("sort_alt",  // lambda function is needed for inlining
              [](epu8 p) {return HPCombi::sorted(p);},
              "netw_lambda", perm16_bench_data.sample);
    return 0;
}


int dummy1 = Bench_sort();

BENCHMARK_MAIN();
