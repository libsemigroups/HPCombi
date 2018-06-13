#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "fonctions_gpu.cuh"

template <typename T>
void shufl_gpu(T* __restrict__ x, const T* __restrict__ y, const size_t size, float * timers);


// Instantiating template functions
template void shufl_gpu<uint8_t>(uint8_t* x, const uint8_t* y, const size_t size, float * timers);
template void shufl_gpu<uint16_t>(uint16_t* x, const uint16_t* y, const size_t size, float * timers);
template void shufl_gpu<uint32_t>(uint32_t* x, const uint32_t* y, const size_t size, float * timers);

// Allocating memory
MemGpu memGpu(131072);

template <typename T>
void shufl_gpu(T* __restrict__ x, const T* __restrict__ y, const size_t size, float * timers)
{
	//Creation des timers	
	cudaEvent_t start_all, stop_all;
	cudaEventCreate(&start_all);
	cudaEventCreate(&stop_all);
	cudaEventRecord(start_all);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaSetDevice(0);
	float tmp=0;

	//~ printf("size : %d\n", size);
	
	// Memory allocation on GPU
	cudaEventRecord(start);
	T *d_x, *d_y, *d_z;

	if (std::is_same<uint8_t, T>::value){
		d_x = (T*)memGpu.d_x8;
		d_y = (T*)memGpu.d_y8;
		d_z = (T*)memGpu.d_z8;
	}
	else if (std::is_same<uint16_t, T>::value){
		d_x = (T*)memGpu.d_x16;
		d_y = (T*)memGpu.d_y16;
		d_z = (T*)memGpu.d_z16;
	}
	else if (std::is_same<uint32_t, T>::value){
		d_x = (T*)memGpu.d_x32;
		d_y = (T*)memGpu.d_y32;
		d_z = (T*)memGpu.d_z32;
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+2, start, stop);

	// Definition of grid and block sizes
	dim3 block(128,1);
	dim3 grid((size + block.x-1)/block.x,1);

	// Copy CPU to GPU
	cudaEventRecord(start);
	gpuErrchk( cudaMemcpy(d_x, x, size*sizeof(T), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_y, y, size*sizeof(T), cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+1, start, stop);
	
		// Computation
		cudaEventRecord(start);
		//~ permute_gpu<T><<<grid, block, block.x*sizeof(T)>>>(d_x, d_y, d_z, size); // Algorithm using sfhl and shared memory
		permute_gpu_gen<T><<<grid, block>>>(d_x, d_y, d_z, size); // Simple algorithm
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(timers, start, stop);
		//~ printf("Computation %.3f ms\n", milliseconds);
	
	//Copy GPU to CPU
	cudaEventRecord(start);
	//~ memset(z, 0, size*sizeof(T));
	gpuErrchk( cudaMemcpy(x, d_z, size*sizeof(T), cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp, start, stop);

	// Update timers
	timers[1] += tmp;
	timers[1] += timers[0];
	timers[2] += timers[1];

}


#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH