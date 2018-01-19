#include <benchmark/benchmark.h>
//~ #include "perm16.hpp"
//~ #include "perm_generic.hpp"
#include "bench_fixture.hpp"

using HPCombi::epu8;
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;
using HPCombi::VectGeneric;

typedef Perm16 ( Perm16::*PERM16_OUT_FUNC ) () const;
typedef uint8_t ( Perm16::*UNINT8_OUT_FUNC ) () const;
typedef Vect1024 ( Vect1024::*COMPOSE_FUNC ) (const Vect1024&) const;
typedef Vect1024 ( Vect1024::*COMPOSE_GPU_FUNC ) (const Vect1024&, float*) const;
 
template<typename T, typename TF> 
void generic_register(benchmark::State& st, const char* label, const std::vector<T> sample, TF pfunc);
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect1024> sample, COMPOSE_FUNC compose_func); 


const Fix_perm16 perm16_bench_data;
const Fix_generic generic_bench_data;

//##################################################################################
  
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect1024> sample, COMPOSE_FUNC compose_func) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  benchmark::DoNotOptimize(
		  (elem.*compose_func)(elem)
		  )
		  ;
	  }
  }
  st.SetLabel(label);
}
  
void compose_gpu_register(benchmark::State& st, const char* label, const std::vector<Vect1024> sample, COMPOSE_GPU_FUNC compose_func, int timer_select) {
  float timers[4] = {0, 0, 0, 0};
  float gpu_timer_total = 0;
  for (auto _ : st) {
	  gpu_timer_total = 0;
	  for (auto elem : sample){
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

template<typename T, typename TF> 
void generic_register(benchmark::State& st, const char* label, const std::vector<T> sample, TF pfunc) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  benchmark::DoNotOptimize(
		  (elem.*pfunc)()
		  )
		  ;
	  }
  }
  st.SetLabel(label);
}

typedef Perm16 ( Perm16::*PERM16_OUT_FUNC ) () const;


//##################################################################################
int RegisterFromFunction_inverse() {
	//~ const float min_time = 0.000001;
	auto REF = benchmark::RegisterBenchmark("inverse_ref", &generic_register<Perm16, PERM16_OUT_FUNC>, "ref", perm16_bench_data.sample, &Perm16::inverse_ref);
    auto ALT_REF = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "ref2", perm16_bench_data.sample, &Perm16::inverse_ref);
    auto ALT_ARR = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "arr", perm16_bench_data.sample, &Perm16::inverse_arr);
    auto ALT_SORT = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "sort", perm16_bench_data.sample, &Perm16::inverse_sort);
    auto ALT_FIND = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "find", perm16_bench_data.sample, &Perm16::inverse_find);
    auto ALT_POW = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "pow", perm16_bench_data.sample, &Perm16::inverse_pow);
    auto ALT_CYCL = benchmark::RegisterBenchmark("inverse_alt", &generic_register<Perm16, PERM16_OUT_FUNC>, "cycl", perm16_bench_data.sample, &Perm16::inverse_cycl);
    
    //~ const std::pair<std::string, const Vect16> vect16s[] = {{"inverse_ref", inverse}, {"inverse_arr", inverse}, {"inverse_sort", inverse}};
	//~ std::vector<benchmark::Benchmark> bnchs = {REF, ALT_REF, ALT_ARR, ALT_SORT, ALT_FIND, ALT_POW, ALT_CYCL};

	//~ REF->Unit(benchmark::kMillisecond); 
	
	//~ REF->MinTime(min_time); 
	//~ ALT_REF->MinTime(min_time); 
	//~ ALT_ARR->MinTime(min_time); 
	//~ ALT_SORT->MinTime(min_time); 
	//~ ALT_FIND->MinTime(min_time); 
	//~ ALT_POW->MinTime(min_time);
	//~ ALT_CYCL->MinTime(min_time);
}

int RegisterFromFunction_compose() {    
    auto REF_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_ref", &compose_register, "ref", generic_bench_data.sample, &Vect1024::permuted);
    auto ALT_COMPOSE_CPU = benchmark::RegisterBenchmark("compose_alt", &compose_register, "cpu", generic_bench_data.sample, &Vect1024::permuted);
    #if COMPILE_CUDA==1
		auto ALT_COMPOSE_GPU = benchmark::RegisterBenchmark("compose_alt", &compose_register, "gpu alloc copy", generic_bench_data.sample, &Vect1024::permuted_gpu);
		for (int timer_select=0; timer_select<4; timer_select++){
			auto ALT_COMPOSE_GPU_TIMER = benchmark::RegisterBenchmark("compose_alt", &compose_gpu_register, "gpu", generic_bench_data.sample, &Vect1024::permuted_gpu_timer, timer_select);
			ALT_COMPOSE_GPU_TIMER->UseManualTime()->MinTime(0.00001);
		}
    #endif  // USE_CUDA
  return 0;
}

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

int dummy0 = RegisterFromFunction_compose();
int dummy2 = RegisterFromFunction_inverse();
int dummy1 = RegisterFromFunction();


BENCHMARK_MAIN();
