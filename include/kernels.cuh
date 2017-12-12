#ifndef HPCOMBI_PERM_KERNELS_CUH
#define HPCOMBI_PERM_KERNELS_CUH

#include <stdint.h>
#include <stdio.h>

template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const size_t Size);


template <typename T>
__global__ void permute_gpu (T * __restrict__ d_x, T * __restrict__ d_y, const size_t Size) {
  // Global thread id and warp id
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t wid = threadIdx.x/warpSize;
  
  // Copy in register
  const T x_reg = d_x[tid];
  const T y_reg = d_y[tid];
  
  // Mask for shuffle greater than warp size
  const bool mask = (y_reg/warpSize == wid) ? 1:0;
  
  // Shared memory for shuffle greater than warp size lesser than block size
  // extern __shared__ T my_shared[]; // Incompatible with use of template
  extern __shared__ __align__(sizeof(T)) unsigned char my_shared[];
  T *shared = reinterpret_cast<T *>(my_shared);
  
  if (tid < Size){
	// Copy in shared memory for shared memory shuffle
	shared[tid] = x_reg;
	
	 // Warp shuffle
	 // y_reg is automaticaly set to y_reg%warpSize 
     d_x[tid] = __shfl(x_reg, y_reg); // Todo try with tmp save in register
										
	// Waitting for all thread to finish shared memory writing
	__syncthreads();	
	//Shared memory shuffle
	if(mask == false){
		d_x[tid] = shared[y_reg];
	}
	
  }
}

#endif  // HPCOMBI_PERM_KERNELS_CUH
