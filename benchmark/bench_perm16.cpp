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

#include "perm16.hpp"
#include "perm_generic.hpp"

using namespace FeatureDetector;
using namespace std;
using HPCombi::epu8;

// const Fix_perm16 sample;
const Fix_epu8 sample;
const std::string SIMDSET = cpu_x86::get_highest_SIMD();
const std::string PROCID = cpu_x86::get_proc_string();

using HPCombi::epu8;
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;

//##################################################################################
template<typename TF, typename Sample>
void myBench(const string &name, TF pfunc, Sample &sample) {
    string fullname = name + "_" + CXX_VER + "_proc-" + PROCID;
    benchmark::RegisterBenchmark(fullname.c_str(),
        [pfunc](benchmark::State& st, Sample &sample) {
            for (auto _ : st) {
                for (auto elem : sample) {
                    for (int i = 0; i < 100; i++)
                        elem = pfunc(elem);
                    benchmark::DoNotOptimize(elem);
                }
            }
        }, sample);
}

#define MYBENCH(nm, fun, smp)  \
    myBench(nm, [](epu8 p) { return fun(p); }, smp)
#define MYBENCH2(nm, fun, smp) \
    myBench(nm, [](epu8 p) { return fun(p,bla); }, smp)


//##################################################################################
int Bench_inverse() {
    myBench("inverse_ref1", [](Perm16 p) { return p.inverse_ref(); }, sample.perms);
    myBench("inverse_ref2", [](Perm16 p) { return p.inverse_ref(); }, sample.perms);
    myBench("inverse_arr", [](Perm16 p) { return p.inverse_arr(); }, sample.perms);
    myBench("inverse_sort", [](Perm16 p) { return p.inverse_sort(); }, sample.perms);
    myBench("inverse_find", [](Perm16 p) { return p.inverse_find(); }, sample.perms);
    myBench("inverse_pow", [](Perm16 p) { return p.inverse_pow(); }, sample.perms);
    myBench("inverse_cycl", [](Perm16 p) { return p.inverse_cycl(); }, sample.perms);
    return 0;
}

/*

int RegisterFromFunction() {
    auto REF_SUM = benchmark::RegisterBenchmark("sum_ref", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::sum_ref);
    auto ALT_SUM_REF = benchmark::RegisterBenchmark("sum_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::sum_ref);
    auto ALT_SUM3 = benchmark::RegisterBenchmark("sum_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "sum3", perm16_bench_data.sample, &Perm16::sum3);
    auto ALT_SUM4 = benchmark::RegisterBenchmark("sum_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "sum4", perm16_bench_data.sample, &Perm16::sum4);

    auto REF_LENGTH = benchmark::RegisterBenchmark("length_ref", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::length_ref);
    auto ALT_LENGTH_REF = benchmark::RegisterBenchmark("length_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::length_ref);
    auto ALT_LENGTH = benchmark::RegisterBenchmark("length_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "length", perm16_bench_data.sample, &Perm16::length);

    auto REF_NB_DESCENT = benchmark::RegisterBenchmark("nb_descent_ref", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::nb_descent_ref);
    auto ALT_NB_DESCENT_REF = benchmark::RegisterBenchmark("nb_descent_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::nb_descent_ref);
    auto ALT_NB_DESCENT = benchmark::RegisterBenchmark("nb_descent_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "nb_descent", perm16_bench_data.sample, &Perm16::nb_descent);

    auto REF_NB_CYCLES = benchmark::RegisterBenchmark("nb_cycles_ref", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::nb_cycles_ref);
    auto ALT_NB_CYCLES_REF = benchmark::RegisterBenchmark("nb_cycles_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::nb_cycles_ref);
    auto ALT_NB_CYCLES_UNROLL = benchmark::RegisterBenchmark("nb_cycles_alt", &generic_register<Perm16, UNINT8_OUT_FUNC>, "unroll", perm16_bench_data.sample, &Perm16::nb_cycles_unroll);

	return 0;
}
*/

auto dummy = {
    Bench_inverse()
};

BENCHMARK_MAIN();
