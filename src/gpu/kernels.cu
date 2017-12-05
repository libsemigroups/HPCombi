#include <stdint.h>

__global__ void permute_gpu (uint8_t *d_x, uint8_t *d_y, const size_t Size) {
  // Global thread id
  int i = blockIdx.x * blockDim.x + threadIdx.x;   
  if (i < Size && i < 32){
    d_x[i] = __shfl(d_x[i], d_y[i], Size);
  }
}
