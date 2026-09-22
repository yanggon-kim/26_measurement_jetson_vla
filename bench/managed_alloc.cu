// T010 / E11b — a PyTorch pluggable allocator backed by cudaMallocManaged.
//
// Protocol §13.3 asks for a `zero_copy` transfer method: "one allocation both CPU and GPU
// address, no copy". On this Tegra the GPU and the CPU share physical LPDDR5, so managed
// memory really is one allocation — there is no copy to time, only the cost of the CPU and the
// GPU touching the same pages.
//
// PyTorch's documented route to foreign memory is
// `torch.cuda.memory.CUDAPluggableAllocator(<this .so>, "managed_malloc", "managed_free")`,
// which must be installed before the process touches CUDA. `bench/xfer_latency.py` therefore
// runs its zero-copy measurements in a dedicated subprocess (`--zero-copy-worker`).
//
// Build (also recorded in env/README.md):
//   nvcc -O3 -arch=sm_87 --shared -Xcompiler -fPIC \
//        -o bench/managed_alloc.so bench/managed_alloc.cu

#include <cuda_runtime.h>
#include <cstdio>

extern "C" {

void *managed_malloc(ssize_t size, int device, cudaStream_t stream) {
  void *ptr = nullptr;
  cudaError_t rc = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
  if (rc != cudaSuccess) {
    std::fprintf(stderr, "managed_malloc(%zd) failed: %s\n", (ssize_t)size,
                 cudaGetErrorString(rc));
    return nullptr;
  }
  return ptr;
}

void managed_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
  cudaFree(ptr);
}

}  // extern "C"
