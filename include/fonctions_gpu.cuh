#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH
#if COMPILE_CUDA==1
	//~ #include <cuda.h>
	//~ #include <helper_cuda.h>
	//~ #include <helper_functions.h>
	#include <cuda_runtime.h>
	template <typename T>
	void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t size, float * timers);

	// GPU error catching
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
	{
	   if (code != cudaSuccess) 
	   {
	      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	      if (abort) exit(code);
	   }
	}
	

	// Memory pool
	class MemGpu {
	
		public :
		uint8_t *d_x8, *d_y8;
		uint16_t *d_x16, *d_y16;
		uint32_t *d_x32, *d_y32;
		int *d_xi, *d_yi;
	
		MemGpu (size_t size) {
			gpuErrchk( cudaMalloc((void**)&d_x8, size*sizeof(uint8_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y8, size*sizeof(uint8_t)) );
	
			gpuErrchk( cudaMalloc((void**)&d_x16, size*sizeof(uint16_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y16, size*sizeof(uint16_t)) );
	
			gpuErrchk( cudaMalloc((void**)&d_x32, size*sizeof(uint32_t)) );
			gpuErrchk( cudaMalloc((void**)&d_y32, size*sizeof(uint32_t)) );
	
			gpuErrchk( cudaMalloc((void**)&d_xi, size*sizeof(int)) );
			gpuErrchk( cudaMalloc((void**)&d_yi, size*sizeof(int)) );
		}
		~MemGpu () {
			gpuErrchk( cudaFree(d_x8) );
			gpuErrchk( cudaFree(d_y8) );
			gpuErrchk( cudaFree(d_x16) );
			gpuErrchk( cudaFree(d_y16) );
			gpuErrchk( cudaFree(d_x32) );
			gpuErrchk( cudaFree(d_y32) );
			gpuErrchk( cudaFree(d_xi) );
			gpuErrchk( cudaFree(d_yi) );
			
		}
	};
	
#endif  // USE_CUDA
#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
