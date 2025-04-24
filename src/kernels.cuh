#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "var.cuh"

__global__ void initDist(
    float * __restrict__ f
);

// ============================================================================================== //

__global__ void gpuPhaseField(
    float * __restrict__ phi,
    const float * __restrict__ g
);

__global__ void gpuGradients(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    float * __restrict__ indicator
);

__global__ void gpuCurvature(
    const float * __restrict__ indicator,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    float * __restrict__ ffx,
    float * __restrict__ ffy,
    float * __restrict__ ffz
);

__global__ void gpuMomOneCollisionStream(
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ rho,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    float * __restrict__ f
);

__global__ void gpuTwoCollisionStream(
    float * __restrict__ g,
    const float * __restrict__ ux,
    const float * __restrict__ uy,
    const float * __restrict__ uz,
    const float * __restrict__ phi,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz
);

__global__ void gpuInflow(
    float * __restrict__ rho,
    float * __restrict__ phi,
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ f,
    float * __restrict__ g,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz
    //const int STEP, const int MACRO_SAVE
);

#endif
