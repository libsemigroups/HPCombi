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

static void escape(epu8 *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
  asm volatile("" : : : "memory");
}

template<typename T>
void use(T &&t) {
  __asm__ __volatile__ ("" :: "g" (t));
}

const Fix_perm16 perm16_bench_data;
const Fix_generic generic_bench_data;

typedef Perm16 ( Perm16::*INVERSE_FUNC ) () const;
typedef uint8_t ( Perm16::*SUM_FUNC ) () const;
typedef Vect1024 ( Vect1024::*COMPOSE_FUNC ) (const Vect1024&) const;
  
  
void inverse_register(benchmark::State& st, const char* label, const std::vector<Perm16> sample, INVERSE_FUNC inverse_func) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  benchmark::DoNotOptimize(
		  (elem.*inverse_func)()
		  )
		  ;
		  escape(&elem.v);
	  }		  
	  clobber();
  }
  st.SetLabel(label);
}
  
void sum_register(benchmark::State& st, const char* label, const std::vector<Perm16> sample, SUM_FUNC sum_func) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  benchmark::DoNotOptimize(
		  (elem.*sum_func)()
		  )
		  ;
		  escape(&elem.v);
	  }		  
	  clobber();
  }
  st.SetLabel(label);
}

  
void compose_register(benchmark::State& st, const char* label, const std::vector<Vect1024> sample, COMPOSE_FUNC compose_func) { 
  for (auto _ : st) {
	  for (auto elem : sample){
		  benchmark::DoNotOptimize(
		  (elem.*compose_func)(elem)
		  )
		  ;
		  //~ escape(&elem.v);
	  }		  
	  clobber();
  }
  st.SetLabel(label);
}



int RegisterFromFunction_inverse() {
	const float min_time = 0.00001;
    auto REF = benchmark::RegisterBenchmark("inverse_ref", &inverse_register, "ref", perm16_bench_data.sample, &Perm16::inverse_ref);
    auto ALT_REF = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "ref", perm16_bench_data.sample, &Perm16::inverse_ref);
    auto ALT_ARR = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "arr", perm16_bench_data.sample, &Perm16::inverse_arr);
    auto ALT_SORT = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "sort", perm16_bench_data.sample, &Perm16::inverse_sort);
    auto ALT_FIND = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "find", perm16_bench_data.sample, &Perm16::inverse_find);
    auto ALT_POW = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "pow", perm16_bench_data.sample, &Perm16::inverse_pow);
    auto ALT_CYCL = benchmark::RegisterBenchmark("inverse_alt", &inverse_register, "cycl", perm16_bench_data.sample, &Perm16::inverse_cycl);	
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
		auto ALT_COMPOSE_GPU = benchmark::RegisterBenchmark("compose_alt", &compose_register, "gpu", generic_bench_data.sample, &Vect1024::permuted_gpu);
    #endif  // USE_CUDA
  return 0;
}

int RegisterFromFunction_sum() {    
    auto REF_SUM = benchmark::RegisterBenchmark("sum_ref", &sum_register, "ref", perm16_bench_data.sample, &Perm16::sum_ref);
    auto ALT_SUM_REF = benchmark::RegisterBenchmark("sum_alt", &sum_register, "ref", perm16_bench_data.sample, &Perm16::sum_ref);
    auto ALT_SUM3 = benchmark::RegisterBenchmark("sum_alt", &sum_register, "sum3", perm16_bench_data.sample, &Perm16::sum3);
    auto ALT_SUM4 = benchmark::RegisterBenchmark("sum_alt", &sum_register, "sum4", perm16_bench_data.sample, &Perm16::sum4);
  return 0;
}

int dummy0 = RegisterFromFunction_compose();
int dummy1 = RegisterFromFunction_sum();
int dummy2 = RegisterFromFunction_inverse();


BENCHMARK_MAIN();
