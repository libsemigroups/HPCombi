#ifndef HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "fonctions_gpu.cuh"

template <typename T>
void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t size, float * timers);


// Instantiating template functions
template void shufl_gpu<uint8_t>(const uint8_t* x, const uint8_t* y, uint8_t* z, const size_t size, float * timers);
template void shufl_gpu<uint16_t>(const uint16_t* x, const uint16_t* y, uint16_t* z, const size_t size, float * timers);
template void shufl_gpu<uint32_t>(const uint32_t* x, const uint32_t* y, uint32_t* z, const size_t size, float * timers);
template void shufl_gpu<int>(const int* x, const int* y, int* z, const size_t size, float * timers);

// Allocating memory
MemGpu memGpu(131072);

template <typename T>
void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t size, float * timers)
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
	T *d_x, *d_y;

	if (std::is_same<uint8_t, T>::value){
		d_x = (T*)memGpu.d_x8;
		d_y = (T*)memGpu.d_y8;
	}
	else if (std::is_same<uint16_t, T>::value){
		d_x = (T*)memGpu.d_x16;
		d_y = (T*)memGpu.d_y16;
	}
	else if (std::is_same<uint32_t, T>::value){
		d_x = (T*)memGpu.d_x32;
		d_y = (T*)memGpu.d_y32;
	}
	else if (std::is_same<int, T>::value){
		d_x = (T*)memGpu.d_xi;
		d_y = (T*)memGpu.d_yi;
	}
	//~ else {	
		//~ cudaMalloc((void**)&d_x, size*sizeof(T));
		//~ cudaMalloc((void**)&d_y, size*sizeof(T));		
	//~ }

	//~ cudaMalloc((void**)&d_x, size*sizeof(T));
	//~ cudaMalloc((void**)&d_y, size*sizeof(T));		

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+2, start, stop);

	// Definition of grid and block sizes
	dim3 block(128,1);
	dim3 grid((size+block.x-1)/block.x,1);

	// Copy CPU to GPU
	cudaEventRecord(start);
	gpuErrchk( cudaMemcpy(d_x, x, size*sizeof(T), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_y, y, size*sizeof(T), cudaMemcpyHostToDevice) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(timers+1, start, stop);
	
		// Computation
		cudaEventRecord(start);
		//~ permute_gpu<T><<<grid, block, block.x*sizeof(T)>>>(d_x, d_y, size); // Algorithm using sfhl and shared memory
		permute_gpu_gen<T><<<grid, block>>>(d_x, d_y, size); // Simple algorithm
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(timers, start, stop);
		//~ printf("Computation %.3f ms\n", milliseconds);
	
	//Copy GPU to CPU
	cudaEventRecord(start);
	gpuErrchk( cudaMemcpy(z, d_y, size*sizeof(T), cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp, start, stop);
	
	// Free GPU memory
	//~ cudaFree(d_x);
	//~ cudaFree(d_y);
	
	cudaEventRecord(stop_all);
	cudaEventSynchronize(stop_all);
	cudaEventElapsedTime(timers+3, start_all, stop_all);
	timers[1] += tmp;
	timers[1] += timers[0];
	timers[2] += timers[1];

}


#endif  // HPCOMBI_PERM_FONCTIONS_GPU_IMPL_CUH
