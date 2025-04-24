#include "kernels.cuh"
#include "globalFunctions.cuh"

__global__ void initDist(float * __restrict__ f) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4D = gpuIdxGlobal4(x,y,z,Q);
        f[idx4D] = W[Q];
    }
}

// =================================================================================================== //

__global__ void gpuPhaseField(
    float * __restrict__ phi,
    const float * __restrict__ g
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx = gpuIdxGlobal3(x,y,z);

    float phiVal = 0.0f;
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4D = gpuIdxGlobal4(x,y,z,Q);
        phiVal += __ldg(&g[idx4D]);
    }

    phi[idx] = phiVal;
}

__global__ void gpuGradients(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    float * __restrict__ indicator
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx = gpuIdxGlobal3(x,y,z);

    float gradx = 3.0f * (W[1]  * phi[IDX3D(x+1,y,z)]   - W[2]  * phi[IDX3D(x-1,y,z)]
                        + W[7]  * phi[IDX3D(x+1,y+1,z)] - W[8]  * phi[IDX3D(x-1,y-1,z)]
                        + W[9]  * phi[IDX3D(x+1,y,z+1)] - W[10] * phi[IDX3D(x-1,y,z-1)]
                        + W[13] * phi[IDX3D(x+1,y-1,z)] - W[14] * phi[IDX3D(x-1,y+1,z)]
                        + W[15] * phi[IDX3D(x+1,y,z-1)] - W[16] * phi[IDX3D(x-1,y,z+1)]);

    float grady = 3.0f * (W[3]  * phi[IDX3D(x,y+1,z)]   - W[4]  * phi[IDX3D(x,y-1,z)]
                        + W[7]  * phi[IDX3D(x+1,y+1,z)] - W[8]  * phi[IDX3D(x-1,y-1,z)]
                        + W[11] * phi[IDX3D(x,y+1,z+1)] - W[12] * phi[IDX3D(x,y-1,z-1)]
                        - W[13] * phi[IDX3D(x+1,y-1,z)] + W[14] * phi[IDX3D(x-1,y+1,z)]
                        + W[17] * phi[IDX3D(x,y+1,z-1)] - W[18] * phi[IDX3D(x,y-1,z+1)]);
    
    float gradz = 3.0f * (W[5]  * phi[IDX3D(x,y,z+1)]   - W[6]  * phi[IDX3D(x,y,z-1)]
                        + W[9]  * phi[IDX3D(x+1,y,z+1)] - W[10] * phi[IDX3D(x-1,y,z-1)]
                        + W[11] * phi[IDX3D(x,y+1,z+1)] - W[12] * phi[IDX3D(x,y-1,z-1)]
                        - W[15] * phi[IDX3D(x+1,y,z-1)] + W[16] * phi[IDX3D(x-1,y,z+1)]
                        - W[17] * phi[IDX3D(x,y+1,z-1)] + W[18] * phi[IDX3D(x,y-1,z+1)]);
    
    float gmagsq = gradx*gradx + grady*grady + gradz*gradz;
    float factor = rsqrtf(fmaxf(gmagsq, 1e-9));

    normx[idx] = gradx * factor;
    normy[idx] = grady * factor;
    normz[idx] = gradz * factor; 
    indicator[idx] = gmagsq * factor;  
}

__global__ void gpuCurvature(
    const float * __restrict__ indicator,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    float * __restrict__ ffx,
    float * __restrict__ ffy,
    float * __restrict__ ffz
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int x = tx + blockIdx.x * blockDim.x;
    const int y = ty + blockIdx.y * blockDim.y;
    const int z = tz + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    const int lx = tx + 1;
    const int ly = ty + 1;
    const int lz = tz + 1;

    __shared__ float s_normx[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];
    __shared__ float s_normy[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];
    __shared__ float s_normz[BLOCK_SIZE_Z+2][BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];

    const int idx = gpuIdxGlobal3(x,y,z);
    
    // load bulk into shared
    s_normx[lz][ly][lx] = normx[idx];
    s_normy[lz][ly][lx] = normy[idx];
    s_normz[lz][ly][lx] = normz[idx];

    // load halos into shared
    if (tx == 0) {
        s_normx[lz][ly][lx-1] = normx[gpuIdxGlobal3(x-1,y,z)];
        s_normy[lz][ly][lx-1] = normy[gpuIdxGlobal3(x-1,y,z)];
        s_normz[lz][ly][lx-1] = normz[gpuIdxGlobal3(x-1,y,z)];
    }
    if (tx == BLOCK_SIZE_X-1) {
        s_normx[lz][ly][lx+1] = normx[gpuIdxGlobal3(x+1,y,z)];
        s_normy[lz][ly][lx+1] = normy[gpuIdxGlobal3(x+1,y,z)];
        s_normz[lz][ly][lx+1] = normz[gpuIdxGlobal3(x+1,y,z)];
    }
    if (ty == 0) {
        s_normx[lz][ly-1][lx] = normx[gpuIdxGlobal3(x,y-1,z)];
        s_normy[lz][ly-1][lx] = normy[gpuIdxGlobal3(x,y-1,z)];
        s_normz[lz][ly-1][lx] = normz[gpuIdxGlobal3(x,y-1,z)];
    }
    if (ty == BLOCK_SIZE_Y-1) {
        s_normx[lz][ly+1][lx] = normx[gpuIdxGlobal3(x,y+1,z)];
        s_normy[lz][ly+1][lx] = normy[gpuIdxGlobal3(x,y+1,z)];
        s_normz[lz][ly+1][lx] = normz[gpuIdxGlobal3(x,y+1,z)];
    }
    if (tz == 0) {
        s_normx[lz-1][ly][lx] = normx[gpuIdxGlobal3(x,y,z-1)];
        s_normy[lz-1][ly][lx] = normy[gpuIdxGlobal3(x,y,z-1)];
        s_normz[lz-1][ly][lx] = normz[gpuIdxGlobal3(x,y,z-1)];
    }
    if (tz == BLOCK_SIZE_Z-1) {
        s_normx[lz+1][ly][lx] = normx[gpuIdxGlobal3(x,y,z+1)];
        s_normy[lz+1][ly][lx] = normy[gpuIdxGlobal3(x,y,z+1)];
        s_normz[lz+1][ly][lx] = normz[gpuIdxGlobal3(x,y,z+1)];
    }
    __syncthreads();

    float curvature = -3.0f * (W[1]  * (CIX[1]  * s_normx[lz][ly][lx+1]   + CIY[1]  * s_normy[lz][ly][lx+1]   + CIZ[1]  * s_normz[lz][ly][lx+1])
                             + W[2]  * (CIX[2]  * s_normx[lz][ly][lx-1]   + CIY[2]  * s_normy[lz][ly][lx-1]   + CIZ[2]  * s_normz[lz][ly][lx-1])
                             + W[3]  * (CIX[3]  * s_normx[lz][ly+1][lx]   + CIY[3]  * s_normy[lz][ly+1][lx]   + CIZ[3]  * s_normz[lz][ly+1][lx])
                             + W[4]  * (CIX[4]  * s_normx[lz][ly-1][lx]   + CIY[4]  * s_normy[lz][ly-1][lx]   + CIZ[4]  * s_normz[lz][ly-1][lx])
                             + W[5]  * (CIX[5]  * s_normx[lz+1][ly][lx]   + CIY[5]  * s_normy[lz+1][ly][lx]   + CIZ[5]  * s_normz[lz+1][ly][lx])
                             + W[6]  * (CIX[6]  * s_normx[lz-1][ly][lx]   + CIY[6]  * s_normy[lz-1][ly][lx]   + CIZ[6]  * s_normz[lz-1][ly][lx])
                             + W[7]  * (CIX[7]  * s_normx[lz][ly+1][lx+1] + CIY[7]  * s_normy[lz][ly+1][lx+1] + CIZ[7]  * s_normz[lz][ly+1][lx+1])
                             + W[8]  * (CIX[8]  * s_normx[lz][ly-1][lx-1] + CIY[8]  * s_normy[lz][ly-1][lx-1] + CIZ[8]  * s_normz[lz][ly-1][lx-1])
                             + W[9]  * (CIX[9]  * s_normx[lz+1][ly][lx+1] + CIY[9]  * s_normy[lz+1][ly][lx+1] + CIZ[9]  * s_normz[lz+1][ly][lx+1])
                             + W[10] * (CIX[10] * s_normx[lz-1][ly][lx-1] + CIY[10] * s_normy[lz-1][ly][lx-1] + CIZ[10] * s_normz[lz-1][ly][lx-1])
                             + W[11] * (CIX[11] * s_normx[lz+1][ly+1][lx] + CIY[11] * s_normy[lz+1][ly+1][lx] + CIZ[11] * s_normz[lz+1][ly+1][lx])
                             + W[12] * (CIX[12] * s_normx[lz-1][ly-1][lx] + CIY[12] * s_normy[lz-1][ly-1][lx] + CIZ[12] * s_normz[lz-1][ly-1][lx])
                             + W[13] * (CIX[13] * s_normx[lz][ly-1][lx+1] + CIY[13] * s_normy[lz][ly-1][lx+1] + CIZ[13] * s_normz[lz][ly-1][lx+1])
                             + W[14] * (CIX[14] * s_normx[lz][ly+1][lx-1] + CIY[14] * s_normy[lz][ly+1][lx-1] + CIZ[14] * s_normz[lz][ly+1][lx-1])
                             + W[15] * (CIX[15] * s_normx[lz-1][ly][lx+1] + CIY[15] * s_normy[lz-1][ly][lx+1] + CIZ[15] * s_normz[lz-1][ly][lx+1])
                             + W[16] * (CIX[16] * s_normx[lz+1][ly][lx-1] + CIY[16] * s_normy[lz+1][ly][lx-1] + CIZ[16] * s_normz[lz+1][ly][lx-1])
                             + W[17] * (CIX[17] * s_normx[lz-1][ly+1][lx] + CIY[17] * s_normy[lz-1][ly+1][lx] + CIZ[17] * s_normz[lz-1][ly+1][lx])
                             + W[18] * (CIX[18] * s_normx[lz+1][ly-1][lx] + CIY[18] * s_normy[lz+1][ly-1][lx] + CIZ[18] * s_normz[lz+1][ly-1][lx]));

    float mult = SIGMA * curvature;
    float indVal = indicator[idx];
    float normxVal = s_normx[lz][ly][lx];
    float normyVal = s_normy[lz][ly][lx];
    float normzVal = s_normz[lz][ly][lx];

    ffx[idx] = mult * normxVal * indVal;
    ffy[idx] = mult * normyVal * indVal;
    ffz[idx] = mult * normzVal * indVal;
}


// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuMomOneCollisionStream(
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ rho,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    float * __restrict__ f
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx = gpuIdxGlobal3(x,y,z);
    
    float fneq[NLINKS];
    float fVal[NLINKS];

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4D = gpuIdxGlobal4(x,y,z,Q);
        fVal[Q] = f[idx4D];
    }

    float rhoVal = fVal[0] + fVal[1] + fVal[2] + fVal[3] + fVal[4] + fVal[5] +
                   fVal[6] + fVal[7] + fVal[8] + fVal[9] + fVal[10] + fVal[11] +
                   fVal[12] + fVal[13] + fVal[14] + fVal[15] + fVal[16] + fVal[17] + fVal[18];

    float invRho = 1.0f / rhoVal;

    float sumUx = invRho * (fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16]);
    float sumUy = invRho * (fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] + fVal[14] - fVal[13] + fVal[17] - fVal[18]);
    float sumUz = invRho * (fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] + fVal[16] - fVal[15] + fVal[18] - fVal[17]);

    float ffxVal = ffx[idx];
    float ffyVal = ffy[idx];
    float ffzVal = ffz[idx];

    float halfFx = ffxVal * 0.5f * invRho;
    float halfFy = ffyVal * 0.5f * invRho;
    float halfFz = ffzVal * 0.5f * invRho;

    float uxVal = sumUx + halfFx;
    float uyVal = sumUy + halfFy;
    float uzVal = sumUz + halfFz;

    float uu = 1.5f * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal);
    float invRhoCssq = 3.0f * invRho;

    float auxHe = 1.0f - OMEGA / 2.0f;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        float pre = gpuFeq(rhoVal,uxVal,uyVal,uzVal,uu,Q);
        float HeF = auxHe * pre * ((CIX[Q] - uxVal) * ffxVal +
                                   (CIY[Q] - uyVal) * ffyVal +
                                   (CIZ[Q] - uzVal) * ffzVal) * invRhoCssq;
        float feq = pre - HeF; 
        fneq[Q] = fVal[Q] - feq;
    }

    float PXX = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7] - fneq[13] + fneq[8] - fneq[14];
    float PXZ = fneq[9] - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];

    ux[idx] = uxVal; uy[idx] = uyVal; uz[idx] = uzVal;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float feq = gpuFeq(rhoVal,uxVal,uyVal,uzVal,uu,Q);
        float HeF = auxHe * feq * ( (CIX[Q] - uxVal) * ffxVal +
                                    (CIY[Q] - uyVal) * ffyVal +
                                    (CIZ[Q] - uzVal) * ffzVal ) * invRhoCssq;
        float fneq = (W[Q] * 4.5f) * gpuTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,Q);
        const int str = gpuIdxGlobal4(xx,yy,zz,Q);
        f[str] = feq + (1.0f - OMEGA) * fneq + HeF; 
    }
}

__global__ void gpuTwoCollisionStream(
    float * __restrict__ g,
    const float * __restrict__ ux,
    const float * __restrict__ uy,
    const float * __restrict__ uz,
    const float * __restrict__ phi,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx = gpuIdxGlobal3(x,y,z);

    float uxVal = ux[idx];
    float uyVal = uy[idx];
    float uzVal = uz[idx];
    float phiVal = phi[idx];
    float normxVal = normx[idx]; 
    float normyVal = normy[idx];
    float normzVal = normz[idx];
    
    float uu = 1.5f * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal);
    float phiNorm = SHARP_C * phiVal * (1.0f - phiVal);
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float geq = gpuFeq(phiVal,uxVal,uyVal,uzVal,uu,Q);
        float Hi = W[Q] * phiNorm * (CIX[Q] * normxVal + CIY[Q] * normyVal + CIZ[Q] * normzVal);
        const int str = gpuIdxGlobal4(xx,yy,zz,Q);
        g[str] = geq + Hi;
    }
}

// =================================================================================================== //



// =================================================================================================== //

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
) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;  

    if (x >= NX || y >= NY) return;

    float cx = NX * 0.5f;
    float cy = NY * 0.5f;

    float dx = x - cx;
    float dy = y - cy;
    float Ri = sqrtf(dx*dx + dy*dy);
    
    if (Ri > DIAM) return;
    float Ri_norm = Ri / DIAM;

    float weight = 1.0f - smoothstep(0.6f, 1.0f, Ri_norm);
    float phiIn = weight;
    float uzIn = U_JET * phiIn; 
    
    const int idxIn = gpuIdxGlobal3(x,y,z);

    float ffxVal = ffx[idxIn];
    float ffyVal = ffy[idxIn];
    float ffzVal = ffz[idxIn];

    float rhoVal = 1.0f;
    float uu = 1.5f * (uzIn * uzIn);
    float auxHe = 1.0f - OMEGA / 2.0f;  

    rho[idxIn] = rhoVal;
    phi[idxIn] = phiIn;
    ux[idxIn] = 0.0f;
    uy[idxIn] = 0.0f;
    uz[idxIn] = uzIn; 

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float feq = gpuFeq(1.0f, 0.0f, 0.0f, uzIn, uu, Q);
        float HeF = auxHe * feq * (CIX[Q] * ffxVal +
                                   CIY[Q] * ffyVal +
                                  (CIZ[Q] - uzIn) * ffzVal) * 3.0f; // was * invRho
        const int str = gpuIdxGlobal4(xx, yy, zz, Q);
        f[str] = feq + HeF;
    }

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float geq = gpuFeq(phiIn, 0.0f, 0.0f, uzIn, uu, Q);
        const int str = gpuIdxGlobal4(xx, yy, zz, Q);
        g[str] = geq;
    }
}

// =================================================================================================== //

