#ifndef GLOBALFUNCTIONS_CUH
#define GLOBALFUNCTIONS_CUH

#include "header.cuh"

//#define FEQSTD

__device__ __forceinline__ int gpuIdxGlobal3(int x, int y, int z) {
    return x + y * NX + z * NX * NY;
}
__device__ __forceinline__ int gpuIdxGlobal4(int x, int y, int z, int Q) {
    int slice = NX * NY;
    return x + y * NX + z * slice + Q * slice * NZ;
}

__device__ __forceinline__ int gpuIdxShared3(int tx, int ty, int tz) {
    return tx + ty * BLOCK_SIZE_X + tz * BLOCK_SIZE_X * BLOCK_SIZE_Y;
}

__device__ __forceinline__ int gpuIdxShared4(int tx, int ty, int tz, int Q) {
    int slice = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    return tx + ty * BLOCK_SIZE_X + tz * slice + Q * slice * BLOCK_SIZE_Z;
}

__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    x = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float gpuTensor2(float XX, float YY, float ZZ, float XY, float XZ, float YZ, int Q) {
    return ((CIX[Q]*CIX[Q] - CSSQ) * XX +
            (CIY[Q]*CIY[Q] - CSSQ) * YY +
            (CIZ[Q]*CIZ[Q] - CSSQ) * ZZ +
            2.0f * CIX[Q] * CIY[Q] * XY +
            2.0f * CIX[Q] * CIZ[Q] * XZ +
            2.0f * CIY[Q] * CIZ[Q] * YZ);
}

__device__ __forceinline__ float gpuFeq(float rho, float ux, float uy, float uz, float uu, int Q) {
    float cu = 3.0f * (ux*CIX[Q] + uy*CIY[Q] + uz*CIZ[Q]);
    float eqbase = rho * (cu + 0.5f * cu*cu - uu);
    return W[Q] * (rho + eqbase);
}

#endif