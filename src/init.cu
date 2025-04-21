#include "kernels.cuh"

__global__ void initDist(
    float * __restrict__ f,
    const int NX, const int NY, const int NZ
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int idx3D = idxGlobal3(x,y,z,NX,NY);

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        int idx4D = idxGlobal4(x,y,z,Q,NX,NY,NZ);
        f[idx4D] = W[Q];
    }
}