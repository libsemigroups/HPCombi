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

#include <cstdlib>
#include <iostream>
#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bench_fixture.hpp"
// #include "compilerinfo.hpp"
// #include "cpu_x86_impl.hpp"

#include "hpcombi/bmat8.hpp"

// using namespace FeatureDetector;
// using namespace std;
// using HPCombi::epu8;

namespace HPCombi {

// const Fix_perm16 sample;
const std::string PROCID = "TODO";

std::vector<BMat8> make_sample(size_t n) {
    std::vector<BMat8> res{};
    for (size_t i = 0; i < n; i++) {
        res.push_back(BMat8::random());
    }
    return res;
}

std::vector<std::pair<BMat8, BMat8>> make_pair_sample(size_t n) {
    std::vector<std::pair<BMat8, BMat8>> res{};
    for (size_t i = 0; i < n; i++) {
        auto x = BMat8::random();
        res.push_back(std::make_pair(x, x));
    }
    return res;
}

class Fix_BMat8 {
  public:
    Fix_BMat8()
        : sample(make_sample(1000)), pair_sample(make_pair_sample(1000)) {}
    ~Fix_BMat8() {}
    const std::vector<BMat8> sample;
    std::vector<std::pair<BMat8, BMat8>>
        pair_sample;  // not const, transpose2 is in place
};

// template <typename TF, typename Sample>
// void myBench(const std::string &name, TF pfunc, Sample &sample) {
//     std::string fullname = name + "_" + CXX_VER + "_proc-" + PROCID;
//     benchmark::RegisterBenchmark(
//         fullname.c_str(), [pfunc, sample](benchmark::State &st) {
//             for (auto _ : st) {
//                 for (auto elem : sample) {
//                     benchmark::DoNotOptimize(pfunc(elem));
//                 }
//             }
//         });
// }

#define BENCHMARK_MEM_FN(mem_fn, sample)                                       \
    BENCHMARK(#mem_fn) {                                                       \
        for (auto &elem : sample) {                                            \
            volatile auto dummy = elem.mem_fn();                               \
        }                                                                      \
        return true;                                                           \
    };

#define BENCHMARK_MEM_FN_PAIR(mem_fn, sample)                                  \
    BENCHMARK(#mem_fn) {                                                       \
        for (auto &pair : sample) {                                            \
            auto val =                                                         \
                std::make_pair(pair.first.mem_fn(), pair.second.mem_fn());     \
            REQUIRE(val.first == val.second);                                  \
        }                                                                      \
        return true;                                                           \
    };

TEST_CASE_METHOD(Fix_BMat8, "Row space size benchmarks 1000 BMat8", "[BMat8][000]") {
    BENCHMARK_MEM_FN(row_space_size_ref, sample);
    BENCHMARK_MEM_FN(row_space_size_bitset, sample);
    BENCHMARK_MEM_FN(row_space_size_incl1, sample);
    BENCHMARK_MEM_FN(row_space_size_incl, sample);
    BENCHMARK_MEM_FN(row_space_size, sample);
}

TEST_CASE_METHOD(Fix_BMat8, "Transpose benchmarks 1000 BMat8", "[BMat8][000]") {
    BENCHMARK_MEM_FN(transpose, sample);
    BENCHMARK_MEM_FN(transpose_mask, sample);
    BENCHMARK_MEM_FN(transpose_maskd, sample);
}

TEST_CASE_METHOD(Fix_BMat8, "Transpose pairs benchmarks 1000 BMat8", "[BMat8][002]") {
    BENCHMARK_MEM_FN_PAIR(transpose, pair_sample);
    BENCHMARK_MEM_FN_PAIR(transpose_mask, pair_sample);
    BENCHMARK_MEM_FN_PAIR(transpose_maskd, pair_sample);
    BENCHMARK("transpose2") {
        for (auto &pair : pair_sample) {
            BMat8::transpose2(pair.first, pair.second);
            REQUIRE(pair.first == pair.second);
        }
        return true;
    };
}
/*



int Bench_row_space_included() {
    myBench(
        "row_space_incl_ref",
        [](std::pair<BMat8, BMat8> p) {
            return p.first.row_space_included_ref(p.second);
        },
        pair_sample);
    myBench(
        "row_space_incl_bitset",
        [](std::pair<BMat8, BMat8> p) {
            return p.first.row_space_included_bitset(p.second);
        },
        pair_sample);
    myBench(
        "row_space_incl_rotate",
        [](std::pair<BMat8, BMat8> p) {
            return p.first.row_space_included(p.second);
        },
        pair_sample);
    return 0;
}

int Bench_row_space_included2() {
    myBench(
        "row_space_incl2_rotate",
        [](std::pair<BMat8, BMat8> p) {
            return p.first.row_space_included(p.second) ==
                   p.second.row_space_included(p.first);
        },
        pair_sample);
    myBench(
        "row_space_incl2",
        [](std::pair<BMat8, BMat8> p) {
            auto res = BMat8::row_space_included2(p.first, p.second,
p.second, p.first); return res.first == res.second;
        },
        pair_sample);
    return 0;
}

*/
}  // namespace HPCombi
