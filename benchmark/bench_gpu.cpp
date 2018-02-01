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
using HPCombi::VectGeneric;

typedef Vect1024 ( Vect1024::*COMPOSE_FUNC ) (const Vect1024&) const;
typedef Vect1024 ( Vect1024::*COMPOSE_GPU_FUNC ) (const Vect1024&, float*) const;
 
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect1024> & sample, COMPOSE_FUNC compose_func); 
void compose_gpu_register(benchmark::State& st, const char* label, const std::vector<Vect1024> & sample, COMPOSE_GPU_FUNC compose_func, int timer_select);
void compose_gpu_one_register(benchmark::State& st, const char* label, const Vect1024 & elem, COMPOSE_GPU_FUNC compose_func, int timer_select);

//const Fix_perm16 perm16_bench_data;
const Fix_generic generic_bench_data;

//##################################################################################
// Register fuction for compose operation : takes one argument
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect1024> & sample, COMPOSE_FUNC compose_func) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  for (int i = 0; i < repeat; i++)
			  benchmark::DoNotOptimize(
			  (elem.*compose_func)(elem)
			  )
			  ;
	  }
  }
  st.SetLabel(label);
}
 
 
//##################################################################################
// Register fuction for compose operation on gpu : uses manual timers
void compose_gpu_register(benchmark::State& st, const char* label, const std::vector<Vect1024> & sample, COMPOSE_GPU_FUNC compose_func, int timer_select) {
  float timers[4] = {0, 0, 0, 0};
  // Computation
  // Computation + copy
  // Computation + copy + allocation
  // Computation + copy + allocation + function call
  float gpu_timer_total = 0;
  for (auto _ : st) {
	  gpu_timer_total = 0;
	  for (auto elem : sample){
		  for (int i = 0; i < repeat; i++) {
			  benchmark::DoNotOptimize(
			  (elem.*compose_func)(elem, timers)
			  )
			  ;
			  gpu_timer_total += timers[timer_select]/1000000;
		  }
	  }
	  st.SetIterationTime(gpu_timer_total);
  }
  st.SetLabel(label);
}
//##################################################################################
// Register fuction for compose operation on gpu : uses manual timers and only one transformation
void compose_gpu_one_register(benchmark::State& st, const char* label, const Vect1024 & elem, COMPOSE_GPU_FUNC compose_func, int timer_select) {
  float timers[4] = {0, 0, 0, 0};
  // Computation
  // Computation + copy
  // Computation + copy + allocation
  // Computation + copy + allocation + function call
  float gpu_timer_total = 0;
  for (auto _ : st) {
	  gpu_timer_total = 0;
	  for (int i = 0; i < repeat; i++) {
		  benchmark::DoNotOptimize(
		  (elem.*compose_func)(elem, timers)
		  )
		  ;
		  gpu_timer_total += timers[timer_select]/1000000;
	  }

	  st.SetIterationTime(gpu_timer_total);
  }
  st.SetLabel(label);
}

//##################################################################################

int RegisterFromFunction_compose() {
    auto REF_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_ref", &compose_register, "ref", generic_bench_data.sample1024, &Vect1024::permuted);
    auto ALT_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_alt", &compose_register, "cpu", generic_bench_data.sample1024, &Vect1024::permuted);
    #if COMPILE_CUDA==1
		auto ALT_COMPOSE_GPU = benchmark::RegisterBenchmark("compose_alt", &compose_register, "gpu function call", generic_bench_data.sample1024, &Vect1024::permuted_gpu);
		std::string gpu = "gpu";
		for (int timer_select=0; timer_select<4; timer_select++){
			auto ALT_COMPOSE_GPU_TIMER = benchmark::RegisterBenchmark("compose_alt", &compose_gpu_register, "gpu", generic_bench_data.sample1024, &Vect1024::permuted_gpu_timer, timer_select);
			ALT_COMPOSE_GPU_TIMER->UseManualTime()->MinTime(0.00005);
		}
		auto ALT_COMPOSE_GPU_SPE_REF = benchmark::RegisterBenchmark("gpu_spe_ref", &compose_gpu_one_register, "ref", generic_bench_data.id, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT0 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register, "id", generic_bench_data.id, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT1 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register, "zeros", generic_bench_data.zeros, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT2 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register, "rand", generic_bench_data.rand, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT3 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register, "randShuf", generic_bench_data.randShuf, &Vect1024::permuted_gpu_timer, 0);

    #endif  // USE_CUDA
  return 0;
}


int dummy0 = RegisterFromFunction_compose();

BENCHMARK_MAIN();
