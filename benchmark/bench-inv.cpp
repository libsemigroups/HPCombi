#include <benchmark/benchmark.h>
//~ #include "perm16.hpp"
//~ #include "perm_generic.hpp"
#include "bench_fixture.hpp"

#include <string.h>
#include <stdlib.h>

using HPCombi::epu8;
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;

const Fix_perm16 perm16_bench_data;

//##################################################################################
// Register fuction for generic operation that take zeros argument
template<typename TF, typename Sample>
void bench_inv(const char* name, TF pfunc, const char* label, Sample sample) {
    benchmark::RegisterBenchmark(name,
        [pfunc, sample](benchmark::State& st, const char* label) {
            for (auto _ : st) {
                for (auto elem : sample) {
                    auto res = elem;
                    for (uint_fast64_t i = 0; i < repeat; i++) res = pfunc(res);
                    benchmark::DoNotOptimize(res);
                }
            }
            st.SetLabel(label);
            st.SetItemsProcessed(st.iterations()*repeat*perm16_bench_data.sample.size());
        }, label);
}



//##################################################################################
int Bench_inverse() {
    bench_inv("inverse_ref", [](Perm16 p) { return p.inverse_ref(); },
              "ref", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_ref(); },
              "ref2", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_arr(); },
              "arr", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_sort(); },
              "sort", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_find(); },
              "find", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_pow(); },
              "pow", perm16_bench_data.sample);
    bench_inv("inverse_alt", [](Perm16 p) { return p.inverse_cycl(); },
              "cycl", perm16_bench_data.sample);
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
// int dummy1 = RegisterFromFunction();

int dummy1 = Bench_inverse();

BENCHMARK_MAIN();
