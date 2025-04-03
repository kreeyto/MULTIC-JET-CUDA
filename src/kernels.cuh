#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "var.cuh"

__global__ void initTensor(
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    float * __restrict__ rho,
    const int NX, const int NY, const int NZ
);

__global__ void initDist(
    const float * __restrict__ rho, 
    const float * __restrict__ phi, 
    float * __restrict__ f,
    float * __restrict__ g,
    const int NX, const int NY, const int NZ
);

// ============================================================================================== //

__global__ void phiCalc(
    float * __restrict__ phi,
    const float * __restrict__ g,
    const int NX, const int NY, const int NZ
);

__global__ void gradCalc(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    float * __restrict__ indicator,
    const int NX, const int NY, const int NZ
);

__global__ void curvatureCalc(
    float * __restrict__ curvature,
    const float * __restrict__ indicator,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    float * __restrict__ ffx,
    float * __restrict__ ffy,
    float * __restrict__ ffz,
    const int NX, const int NY, const int NZ
);

__global__ void momentiCalc(
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ rho,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    const float * __restrict__ f,
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    const int NX, const int NY, const int NZ
);

__global__ void collisionFluid(
    float * __restrict__ f,
    const float * __restrict__ ux,
    const float * __restrict__ uy,
    const float * __restrict__ uz,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    const float * __restrict__ rho,
    const float * __restrict__ pxx,
    const float * __restrict__ pyy,
    const float * __restrict__ pzz,
    const float * __restrict__ pxy,
    const float * __restrict__ pxz,
    const float * __restrict__ pyz,
    const int NX, const int NY, const int NZ
);

__global__ void collisionPhase(
    float * __restrict__ g,
    const float * __restrict__ ux,
    const float * __restrict__ uy,
    const float * __restrict__ uz,
    const float * __restrict__ phi,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    const int NX, const int NY, const int NZ
);

__global__ void fgBoundary(
    float * __restrict__ rho,
    float * __restrict__ phi,
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ f,
    float * __restrict__ g,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    const float U_MAX, const int D_HALF,
    const int NX, const int NY, const int NZ,
    const int STEP, const int MACRO_SAVE
);

#endif
