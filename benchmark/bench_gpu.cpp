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


typedef Vect1024 ( Vect1024::*COMPOSE_FUNC1024 ) (const Vect1024&) const;
typedef Vect1024 ( Vect1024::*COMPOSE_GPU_FUNC1024 ) (const Vect1024&, float*) const;
typedef Vect131072 ( Vect131072::*COMPOSE_FUNC131072 ) (const Vect131072&) const;
typedef Vect131072 ( Vect131072::*COMPOSE_GPU_FUNC131072 ) (const Vect131072&, float*) const;

template<typename Vect, typename Func>
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect> & sample, Func compose_func);
template<typename Vect, typename Func>
void compose_gpu_register(benchmark::State& st, const char* label, const std::vector<Vect> & sample, Func compose_func, int timer_select);
template<typename Vect, typename Func> 
void compose_gpu_one_register(benchmark::State& st, const char* label, const Vect & elem, Func compose_func, int timer_select);

//const Fix_perm16 perm16_bench_data;
const Fix_generic generic_bench_data;

//##################################################################################
// Register fuction for compose operation : takes one argument
template<typename Vect, typename Func> 
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect> & sample, Func compose_func) { 
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
template<typename Vect, typename Func> 
void compose_gpu_register(benchmark::State& st, const char* label, const std::vector<Vect> & sample, Func compose_func, int timer_select) {
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
template<typename Vect, typename Func> 
void compose_gpu_one_register(benchmark::State& st, const char* label, const Vect & elem, Func compose_func, int timer_select) {
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
    auto REF_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_ref", &compose_register<Vect1024, COMPOSE_FUNC1024>, "ref", generic_bench_data.sample1024, &Vect1024::permuted);
    auto ALT_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_alt", &compose_register<Vect1024, COMPOSE_FUNC1024>, "cpu", generic_bench_data.sample1024, &Vect1024::permuted);
    #if COMPILE_CUDA==1
		auto ALT_COMPOSE_GPU = benchmark::RegisterBenchmark("compose_alt", &compose_register<Vect1024, COMPOSE_FUNC1024>, "gpu function call", generic_bench_data.sample1024, &Vect1024::permuted_gpu);
		std::string gpu = "gpu";
		for (int timer_select=0; timer_select<4; timer_select++){
			auto ALT_COMPOSE_GPU_TIMER = benchmark::RegisterBenchmark("compose_alt", &compose_gpu_register<Vect1024, COMPOSE_GPU_FUNC1024>, "gpu", generic_bench_data.sample1024, &Vect1024::permuted_gpu_timer, timer_select);
			ALT_COMPOSE_GPU_TIMER->UseManualTime()->MinTime(0.00005);
		}
		auto ALT_COMPOSE_GPU_SPE_REF = benchmark::RegisterBenchmark("gpu_spe_ref", &compose_gpu_one_register<Vect1024, COMPOSE_GPU_FUNC1024>, "ref", generic_bench_data.id, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT0 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register<Vect1024, COMPOSE_GPU_FUNC1024>, "id", generic_bench_data.id, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT1 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register<Vect1024, COMPOSE_GPU_FUNC1024>, "zeros", generic_bench_data.zeros, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT2 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register<Vect1024, COMPOSE_GPU_FUNC1024>, "rand", generic_bench_data.rand, &Vect1024::permuted_gpu_timer, 0);
		auto ALT_COMPOSE_GPU_SPE_ALT3 = benchmark::RegisterBenchmark("gpu_spe_alt", &compose_gpu_one_register<Vect1024, COMPOSE_GPU_FUNC1024>, "randShuf", generic_bench_data.randShuf, &Vect1024::permuted_gpu_timer, 0);

    #endif  // USE_CUDA
  return 0;
}

int RegisterFromFunction_compose131072() {
    auto REF_COMPOSE_CPU = benchmark::RegisterBenchmark("compose131072_ref", &compose_register<Vect131072, COMPOSE_FUNC131072>, "ref", generic_bench_data.sample131072, &Vect131072::permuted);
    auto ALT_COMPOSE_CPU = benchmark::RegisterBenchmark("compose131072_alt", &compose_register<Vect131072, COMPOSE_FUNC131072>, "cpu", generic_bench_data.sample131072, &Vect131072::permuted);
    #if COMPILE_CUDA==1
		auto ALT_COMPOSE_GPU = benchmark::RegisterBenchmark("compose131072_alt", &compose_register<Vect131072, COMPOSE_FUNC131072>, "gpu function call", generic_bench_data.sample131072, &Vect131072::permuted_gpu);
		std::string gpu = "gpu";
		for (int timer_select=0; timer_select<4; timer_select++){
			auto ALT_COMPOSE_GPU_TIMER = benchmark::RegisterBenchmark("compose131072_alt", &compose_gpu_register<Vect131072, COMPOSE_GPU_FUNC131072>, "gpu", generic_bench_data.sample131072, &Vect131072::permuted_gpu_timer, timer_select);
			ALT_COMPOSE_GPU_TIMER->UseManualTime()->MinTime(0.00005);
		}

    #endif  // USE_CUDA
  return 0;
}


int dummy0 = RegisterFromFunction_compose();
int dummy1 = RegisterFromFunction_compose131072();

BENCHMARK_MAIN();
