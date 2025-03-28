#include "kernels.cuh"

__global__ void initTensor(
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    float * __restrict__ rho,
    const int NX, const int NY, const int NZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float val = 1.0f;
    pxx[idx3D] = val; pyy[idx3D] = val; pzz[idx3D] = val;
    pxy[idx3D] = val; pxz[idx3D] = val; pyz[idx3D] = val;
    rho[idx3D] = val;
}

__global__ void initDist(
    const float * __restrict__ rho, 
    const float * __restrict__ phi, 
    float * __restrict__ f,
    float * __restrict__ g,
    const int NX, const int NY, const int NZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float rho_val = rho[idx3D];
    float phi_val = phi[idx3D];

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        f[idx4D] = W[l] * rho_val;
    }
    
    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        g[idx4D] = W[l] * phi_val;
    }
}