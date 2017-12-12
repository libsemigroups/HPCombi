#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>

template <typename T>
void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t Size);


//Instantiating template functions
template void shufl_gpu<uint8_t>(const uint8_t* x, const uint8_t* y, uint8_t* z, const size_t Size);
template void shufl_gpu<uint16_t>(const uint16_t* x, const uint16_t* y, uint16_t* z, const size_t Size);

template <typename T>
void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t Size)
{
	cudaSetDevice(0);
	//Creation des timers	
	//~ cudaEvent_t start, stop;
	//~ cudaEventCreate(&start);
	//~ cudaEventCreate(&stop);
	//~ float milliseconds = 0;

	//~ printf("Size : %d\n", Size);
	
	// Memory allocation on GPU
	T *d_x, *d_y;
	cudaMalloc((void**)&d_x, Size*sizeof(T));
	cudaMalloc((void**)&d_y, Size*sizeof(T));

	// Definition of grid and block sizes
	dim3 block(Size,1);
	dim3 grid((Size+block.x-1)/block.x,1);

	// Copy CPU to GPU
	cudaMemcpy(d_x, x, Size*sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, Size*sizeof(T), cudaMemcpyHostToDevice);
	
		// Computation
		//~ cudaEventRecord(start);
		permute_gpu<T><<<grid, block, Size*sizeof(T)>>>(d_x, d_y, Size);
		//~ cudaEventRecord(stop);	
		//~ cudaEventSynchronize(stop);
		//~ cudaEventElapsedTime(&milliseconds, start, stop);
		//~ printf("Computation %.3f ms\n", milliseconds);
	
	//Copy GPU to CPU
	cudaMemcpy(z, d_x, Size*sizeof(T), cudaMemcpyDeviceToHost);
	
	// Free GPU memory
	cudaFree(d_x);
	cudaFree(d_y);
}


#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
