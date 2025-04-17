#include "kernels.cuh"

__global__ void initDist(
    float * __restrict__ f,
    const int NX, const int NY, const int NZ
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        f[idx4D] = W[l];
    }
}