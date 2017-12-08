#ifndef HPCOMBI_PERM_FONCTIONS_GPU_CUH
#define HPCOMBI_PERM_FONCTIONS_GPU_CUH

template <typename T>
void shufl_gpu(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ z, const size_t Size);

#endif  // HPCOMBI_PERM_FONCTIONS_GPU_CUH
