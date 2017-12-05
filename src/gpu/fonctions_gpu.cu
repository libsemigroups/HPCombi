#include <stdio.h>
#include <stdint.h>
#include "kernels.cuh"

void shufl_gpu(const uint8_t* x, const uint8_t* y, uint8_t* z, const size_t Size)
{
	// Memory allocation on GPU
	uint8_t *d_x, *d_y;
	cudaMalloc((void**)&d_x, Size*sizeof(uint8_t));
	cudaMalloc((void**)&d_y, Size*sizeof(uint8_t));

	// Definition of grid and block sizes
	dim3 block(Size,1);
	dim3 grid((Size+block.x-1)/block.x,1);

	// Copy CPU to GPU
	cudaMemcpy(d_x, x, Size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, Size*sizeof(uint8_t), cudaMemcpyHostToDevice);
	
		// Computation
		permute_gpu<<<grid, block>>>(d_x, d_y, Size);
	
	//Copy GPU to CPU
	cudaMemcpy(z, d_x, Size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	
	// Free GPU memory
	cudaFree(d_x);
	cudaFree(d_y);
}
