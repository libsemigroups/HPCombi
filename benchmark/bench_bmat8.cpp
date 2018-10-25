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

#include <iostream>
#include <benchmark/benchmark.h>
#include <string.h>
#include <stdlib.h>

#include "compilerinfo.hpp"
#include "cpu_x86_impl.hpp"
#include "bench_fixture.hpp"

#include "bmat8.hpp"

using namespace FeatureDetector;
using namespace std;
using HPCombi::epu8;

// const Fix_perm16 sample;
const std::string SIMDSET = cpu_x86::get_highest_SIMD();
const std::string PROCID = cpu_x86::get_proc_string();

using namespace HPCombi;

std::vector<BMat8> make_sample(size_t n) {
    std::vector<BMat8> res {};
    for (size_t i=0; i < n; i++) {
        res.push_back(BMat8::random());
    }
    return res;
}

std::vector<BMat8> sample = make_sample(1000);
// std::vector<BMat8> sample = {BMat8::one()};
// std::vector<BMat8> sample = {BMat8(0)};

//##################################################################################
template<typename TF, typename Sample>
void myBench(const string &name, TF pfunc, Sample &sample) {
    string fullname = name + "_" + CXX_VER + "_proc-" + PROCID;
    benchmark::RegisterBenchmark(fullname.c_str(),
       [pfunc, sample](benchmark::State& st) {
            for (auto _ : st) {
                for (auto elem : sample) {
                    benchmark::DoNotOptimize(pfunc(elem));
                }
            }
        });
}

#define myBenchMeth(descr, methname, smp) \
    myBench(descr, [](BMat8 p) { return p.methname(); }, smp)

//##################################################################################
int Bench_row_space_size() {
    myBenchMeth("row_space_size_ref", row_space_size_ref, sample);
    myBenchMeth("row_space_size_bitset", row_space_size_bitset, sample);
    myBenchMeth("row_space_size_incl", row_space_size_incl, sample);
    myBenchMeth("row_space_size", row_space_size, sample);
    return 0;
}

auto dummy = {
    Bench_row_space_size()
};

BENCHMARK_MAIN();
