#if COMPILE_CUDA==1
//~ #include <benchmark/benchmark.h>
#include <iostream>
#include "bench_gpu_fixture.hpp"

#include <string.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>

using HPCombi::VectGeneric;


int main(){
	cudaSetDevice(0);
	cudaProfilerStart();
	const Fix_generic generic_bench_data;
	for(int i=0; i<1000; i++)
		generic_bench_data.sample131072[0].permuted_gpu(generic_bench_data.sample131072[0]);	
	cudaProfilerStop();
	return 0;
}


#endif  // USE_CUDA
