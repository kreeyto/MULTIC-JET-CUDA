#include "kernels.cuh"

// ================================================================================================== //

__global__ void gpuPhaseField(
    float * __restrict__ phi,
    const float * __restrict__ g,
    const int NX, const int NY, const int NZ
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    int idx = inline3D(x,y,z,NX,NY);
    float phiVal = 0.0f;

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {
        int idx4D = inline4D(x,y,z,Q,NX,NY,NZ);
        phiVal += g[idx4D];
    }

    phi[idx] = phiVal;
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuGradients(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    float * __restrict__ indicator,
    const int NX, const int NY, const int NZ
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    int idx = inline3D(x,y,z,NX,NY);

    float gradx = 0.0f, grady = 0.0f, gradz = 0.0f;
    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );

        int offset = inline3D(xx,yy,zz,NX,NY);
        float phiVal = phi[offset];
        float coef = 3.0f * W[Q];
        gradx += coef * CIX[Q] * phiVal;
        grady += coef * CIY[Q] * phiVal;
        gradz += coef * CIZ[Q] * phiVal;
    }
    
    float gmagsq = gradx*gradx + grady*grady + gradz*gradz;
    float factor = rsqrtf(fmaxf(gmagsq, 1e-9));

    normx[idx] = gradx * factor;
    normy[idx] = grady * factor;
    normz[idx] = gradz * factor; 
    indicator[idx] = gmagsq * factor;  
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuCurvature(
    const float * __restrict__ indicator,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    float * __restrict__ ffx,
    float * __restrict__ ffy,
    float * __restrict__ ffz,
    const int NX, const int NY, const int NZ
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    int idx = inline3D(x,y,z,NX,NY);

    float normxVal = normx[idx];
    float normyVal = normy[idx];
    float normzVal = normz[idx];
    float indVal = indicator[idx];
    float curvature = 0.0f;

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );

        int offset = inline3D(xx,yy,zz,NX,NY);
        float nox = normx[offset];
        float noy = normy[offset];
        float noz = normz[offset];
        float coef = 3.0f * W[Q];
        curvature -= coef * (CIX[Q]*nox + CIY[Q]*noy + CIZ[Q]*noz);
    }

    float mult = SIGMA * curvature;

    ffx[idx] = mult * normxVal * indVal;
    ffy[idx] = mult * normyVal * indVal;
    ffz[idx] = mult * normzVal * indVal;
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuMomCollisionStream(
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ rho,
    const float * __restrict__ ffx,
    const float * __restrict__ ffy,
    const float * __restrict__ ffz,
    float * __restrict__ f,
    const int NX, const int NY, const int NZ
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    int idx = inline3D(x,y,z,NX,NY);
    
    float fneq[NLINKS];
    float fVal[NLINKS];

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {
        int idx4D = inline4D(x,y,z,Q,NX,NY,NZ);
        fVal[Q] = f[idx4D];
    }

    float rhoVal = 0.0f;
    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) 
        rhoVal += fVal[Q];

    float invRho = 1.0f / rhoVal;

    float sumUx = invRho * (fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16]);
    float sumUy = invRho * (fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] + fVal[14] - fVal[13] + fVal[17] - fVal[18]);
    float sumUz = invRho * (fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] + fVal[16] - fVal[15] + fVal[18] - fVal[17]);

    float ffxVal = ffx[idx];
    float ffyVal = ffy[idx];
    float ffzVal = ffz[idx];

    // the factor 1/2 here emerges from he et al. forcing scheme, where:
    // u = 1/rho * sum_i(c_i*f_i) + A * F/rho
    // thus, with A = 1/2:
    // u = 1/rho * sum_i(c_i*f_i) + F/(2*rho)
    float halfFx = ffxVal * 0.5f * invRho;
    float halfFy = ffyVal * 0.5f * invRho;
    float halfFz = ffzVal * 0.5f * invRho;

    float uxVal = sumUx + halfFx;
    float uyVal = sumUy + halfFy;
    float uzVal = sumUz + halfFz;

    float uu = 1.5f * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal);
    float invRhoCssq = 3.0f / rhoVal;

    float auxHe = 1.0f - OMEGA / 2.0f;

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {
        float cu = 3.0f * (uxVal * CIX[Q] + uyVal * CIY[Q] + uzVal * CIZ[Q]);
        float eqBase = rhoVal * (cu + 0.5f * cu*cu - uu);
        float common = W[Q] * (rhoVal + eqBase);
        float HeF = auxHe * common * ((CIX[Q] - uxVal) * ffxVal +
                                      (CIY[Q] - uyVal) * ffyVal +
                                      (CIZ[Q] - uzVal) * ffzVal) * invRhoCssq;
        float feq = common - HeF; 
        fneq[Q] = fVal[Q] - feq;
    }

    float PXX = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7] - fneq[13] + fneq[8] - fneq[14];
    float PXZ = fneq[9] - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];

    ux[idx] = uxVal; uy[idx] = uyVal; uz[idx] = uzVal;

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );

        float cu = 3.0f * (uxVal*CIX[Q] + uyVal*CIY[Q] + uzVal*CIZ[Q]);
        float feq = W[Q] * (rhoVal + rhoVal * (cu + 0.5f * cu*cu - uu));
        float HeF = auxHe * feq * ( (CIX[Q] - uxVal) * ffxVal +
                                    (CIY[Q] - uyVal) * ffyVal +
                                    (CIZ[Q] - uzVal) * ffzVal ) * invRhoCssq;
        float fneq = (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * PXX +
                                      (CIY[Q]*CIY[Q] - CSSQ) * PYY +
                                      (CIZ[Q]*CIZ[Q] - CSSQ) * PZZ +
                                       2.0f * CIX[Q] * CIY[Q] * PXY +
                                       2.0f * CIX[Q] * CIZ[Q] * PXZ +
                                       2.0f * CIY[Q] * CIZ[Q] * PYZ
                                     );
        int offset = inline4D(xx,yy,zz,Q,NX,NY,NZ);
        f[offset] = feq + (1.0f - OMEGA) * fneq + HeF; 
    }
}

__global__ void gpuPhaseCollisionStream(
    float * __restrict__ g,
    const float * __restrict__ ux,
    const float * __restrict__ uy,
    const float * __restrict__ uz,
    const float * __restrict__ phi,
    const float * __restrict__ normx,
    const float * __restrict__ normy,
    const float * __restrict__ normz,
    const int NX, const int NY, const int NZ
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    int idx = inline3D(x,y,z,NX,NY);

    float uxVal = ux[idx];
    float uyVal = uy[idx];
    float uzVal = uz[idx];
    float phiVal = phi[idx];
    float normxVal = normx[idx]; 
    float normyVal = normy[idx];
    float normzVal = normz[idx];
    
    float uu = 1.5f * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal);
    float phiNorm = SHARP_C * phiVal * (1.0f - phiVal);
    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );

        float cu = 3.0f * (uxVal * CIX[Q] + uyVal * CIY[Q] + uzVal * CIZ[Q]);
        // was using first order
        //float geq = W[Q] * phiVal * (1.0f + cu);
        float geq = W[Q] * (phiVal + phiVal * (cu + 0.5f * cu*cu - uu));
        float Hi = W[Q] * phiNorm * (CIX[Q] * normxVal + CIY[Q] * normyVal + CIZ[Q] * normzVal);
        int offset = inline4D(xx,yy,zz,Q,NX,NY,NZ);
        g[offset] = geq + Hi;
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
    const float * __restrict__ ffz,
    const float U_JET, const int DIAM,
    const int NX, const int NY, const int NZ
    //const int STEP, const int MACRO_SAVE
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (z != 0) return; 
    
    float cx = NX * 0.5f;
    float cy = NY * 0.5f;

    float dx = x - cx;
    float dy = y - cy;
    float Ri = sqrtf(dx*dx + dy*dy);
    
    if (Ri > DIAM) return;

    float phiIn = 1.0f; //0.5f + 0.5f * tanh(2.0f * (DIAM - Ri) / INTERFACE_WIDTH);
    float uzIn = U_JET * phiIn; 
    
    int idxIn = inline3D(x,y,z,NX,NY);

    float ffxVal = ffx[idxIn];
    float ffyVal = ffy[idxIn];
    float ffzVal = ffz[idxIn];

    float rhoVal = 1.0f;
    float uu = 1.5f * (uzIn * uzIn);
    float invRhoCssq = 3.0f / rhoVal;
    float auxHe = 1.0f - OMEGA / 2.0f;  

    rho[idxIn] = rhoVal;
    phi[idxIn] = phiIn;
    ux[idxIn] = 0.0f;
    uy[idxIn] = 0.0f;
    uz[idxIn] = uzIn; 

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {
        float cu = 3.0f * uzIn * CIZ[Q];
        float feq = W[Q] * (1.0f + (cu + 0.5f * cu*cu - uu));
        float HeF = auxHe * feq * (CIX[Q] * ffxVal +
                                   CIY[Q] * ffyVal +
                                   (CIZ[Q] - uzIn) * ffzVal) * invRhoCssq;

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );
        
        int offset = inline4D(xx,yy,zz,Q,NX,NY,NZ);
        f[offset] = feq + HeF;
    }

    #pragma unroll 19
    for (int Q = 0; Q < NLINKS; ++Q) {
        float cu = 3.0f * uzIn * CIZ[Q];
        // was using first order
        //float geq = W[Q] * phiIn * (1.0f + cu);
        float geq = W[Q] * (phiIn + phiIn * (cu + 0.5f * cu*cu - uu));

        int xx = (x + CIX[Q] + NX) % NX;
        int yy = (y + CIY[Q] + NY) % NY;
        int zz = z + CIZ[Q];
        zz = min( max(zz, 1), NZ-2 );

        int offset = inline4D(xx,yy,zz,Q,NX,NY,NZ);
        g[offset] = geq;
    }
}

// =================================================================================================== //

