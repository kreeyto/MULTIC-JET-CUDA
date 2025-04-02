#include "kernels.cuh"

// ================================================================================================== //

__global__ void phiCalc(
    float * __restrict__ phi,
    const float * __restrict__ g,
    const int NX, const int NY, const int NZ
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx = inline3D(i,j,k,NX,NY);

    float phi_local = 0.0f;
    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        phi_local += g[idx4D];
    }
    phi[idx] = phi_local;
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gradCalc(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    float * __restrict__ indicator,
    const int NX, const int NY, const int NZ
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int ii = i + CIX[l];
        int jj = j + CIY[l];
        int kk = k + CIZ[l];

        int offset = inline3D(ii,jj,kk,NX,NY);
        float val = phi[offset];
        float coef = 3.0f * W[l];
        grad_x += coef * CIX[l] * val;
        grad_y += coef * CIY[l] * val;
        grad_z += coef * CIZ[l] * val;
    }
    
    float gmag_sq = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
    float factor = rsqrtf(fmaxf(gmag_sq, 1e-9));

    normx[idx3D] = grad_x * factor;
    normy[idx3D] = grad_y * factor;
    normz[idx3D] = grad_z * factor; 
    indicator[idx3D] = gmag_sq * factor;  
}

// =================================================================================================== //



// =================================================================================================== //

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
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float normx_ = normx[idx3D];
    float normy_ = normy[idx3D];
    float normz_ = normz[idx3D];
    float ind_ = indicator[idx3D];
    float curv = 0.0f;

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int ii = i + CIX[l];
        int jj = j + CIY[l];
        int kk = k + CIZ[l];

        int offset = inline3D(ii,jj,kk,NX,NY);
        float nox = normx[offset];
        float noy = normy[offset];
        float noz = normz[offset];
        float coef = 3.0f * W[l];
        curv -= coef * (CIX[l] * nox + CIY[l] * noy + CIZ[l] * noz);
    }

    float mult = SIGMA * curv;

    curvature[idx3D] = curv;
    ffx[idx3D] = mult * normx_ * ind_;
    ffy[idx3D] = mult * normy_ * ind_;
    ffz[idx3D] = mult * normz_ * ind_;
}

// =================================================================================================== //



// =================================================================================================== //

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
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);
    
    float fneq[NLINKS];
    float fVal[NLINKS];

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        fVal[l] = f[idx4D];
    }

    float rhoVal = 0.0f;
    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) 
        rhoVal += fVal[l];

    float invRho = 1.0f / rhoVal;

    float sumUx = invRho * (fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16]);
    float sumUy = invRho * (fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] + fVal[14] - fVal[13] + fVal[17] - fVal[18]);
    float sumUz = invRho * (fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] + fVal[16] - fVal[15] + fVal[18] - fVal[17]);

    float ffx_val = ffx[idx3D];
    float ffy_val = ffy[idx3D];
    float ffz_val = ffz[idx3D];

    // the factor 1/2 here emerges from he et al. forcing scheme, where:
    // u = 1/rho * sum_i(c_i*f_i) + A * F/rho
    // thus, with A = 1/2:
    // u = 1/rho * sum_i(c_i*f_i) + F/(2*rho)
    float halfFx = ffx_val * 0.5f * invRho;
    float halfFy = ffy_val * 0.5f * invRho;
    float halfFz = ffz_val * 0.5f * invRho;

    float uxVal = sumUx + halfFx;
    float uyVal = sumUy + halfFy;
    float uzVal = sumUz + halfFz;

    float invCssq = 1.0f / CSSQ;
    float uu = 0.5f * (uxVal * uxVal + uyVal * uyVal + uzVal * uzVal) * invCssq;
    float invRhoCssq = 1.0f / (rhoVal * CSSQ);

    float sumXX = 0.0f, sumYY = 0.0f, sumZZ = 0.0f;
    float sumXY = 0.0f, sumXZ = 0.0f, sumYZ = 0.0f;

    float auxHe = 1.0f - OMEGA / 2.0f;

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        float udotc = (uxVal * CIX[l] + uyVal * CIY[l] + uzVal * CIZ[l]) * invCssq;
        float eqBase = rhoVal * (udotc + 0.5f * udotc*udotc - uu);
        float common = W[l] * (rhoVal + eqBase);
        float HeF = auxHe * common * ((CIX[l] - uxVal) * ffx_val +
                                       (CIY[l] - uyVal) * ffy_val +
                                       (CIZ[l] - uzVal) * ffz_val) * invRhoCssq;
        float feq = common - HeF; 
        fneq[l] = fVal[l] - feq;
    }

    sumXX = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    sumYY = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    sumZZ = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    sumXY = fneq[7] - fneq[13] + fneq[8] - fneq[14];
    sumXZ = fneq[9] - fneq[15] + fneq[10] - fneq[16];
    sumYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];

    pxx[idx3D] = sumXX; pyy[idx3D] = sumYY; pzz[idx3D] = sumZZ;
    pxy[idx3D] = sumXY; pxz[idx3D] = sumXZ; pyz[idx3D] = sumYZ;

    ux[idx3D] = uxVal; uy[idx3D] = uyVal; uz[idx3D] = uzVal;
    rho[idx3D] = rhoVal;
}

// =================================================================================================== //



// =================================================================================================== //

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
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float ux_val = ux[idx3D], uy_val = uy[idx3D], uz_val = uz[idx3D], rho_val = rho[idx3D];
    float ffx_val = ffx[idx3D], ffy_val = ffy[idx3D], ffz_val = ffz[idx3D];
    float pxx_val = pxx[idx3D], pyy_val = pyy[idx3D], pzz_val = pzz[idx3D];
    float pxy_val = pxy[idx3D], pxz_val = pxz[idx3D], pyz_val = pyz[idx3D];

    float uu = 0.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val) / CSSQ;
    float invRhoCssq = 1.0f / (rho_val * CSSQ);
    float invCssq = 1.0f / CSSQ;

    float auxHe = 1.0f - OMEGA / 2.0f;

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int ii = i + CIX[l]; 
        int jj = j + CIY[l]; 
        int kk = k + CIZ[l];
        
        float udotc = (ux_val * CIX[l] + uy_val * CIY[l] + uz_val * CIZ[l]) * invCssq;
        float feq = W[l] * (rho_val + rho_val * (udotc + 0.5f * udotc*udotc - uu));
        float HeF = auxHe * feq * ( (CIX[l] - ux_val) * ffx_val +
                                    (CIY[l] - uy_val) * ffy_val +
                                    (CIZ[l] - uz_val) * ffz_val ) * invRhoCssq;
        float fneq = (W[l] / (2.0f * CSSQ * CSSQ)) * ((CIX[l]*CIX[l] - CSSQ) * pxx_val +
                                                      (CIY[l]*CIY[l] - CSSQ) * pyy_val +
                                                      (CIZ[l]*CIZ[l] - CSSQ) * pzz_val +
                                                       2.0f * CIX[l] * CIY[l] * pxy_val +
                                                       2.0f * CIX[l] * CIZ[l] * pxz_val +
                                                       2.0f * CIY[l] * CIZ[l] * pyz_val
                                                    );
        int offset = inline4D(ii,jj,kk,l,NX,NY,NZ);
        f[offset] = feq + (1.0f - OMEGA) * fneq + HeF; 
    }
}

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
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float ux_val = ux[idx3D];
    float uy_val = uy[idx3D];
    float uz_val = uz[idx3D];
    float phi_val = phi[idx3D];
    float normx_val = normx[idx3D]; 
    float normy_val = normy[idx3D];
    float normz_val = normz[idx3D];

    float invCSSQ = 1.0f / CSSQ;

    float phi_norm = SHARP_C * phi_val * (1.0f - phi_val);
    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int ii = i + CIX[l]; 
        int jj = j + CIY[l]; 
        int kk = k + CIZ[l];
        
        float udotc = (ux_val * CIX[l] + uy_val * CIY[l] + uz_val * CIZ[l]) * invCSSQ;
        float geq = W[l] * phi_val * (1.0f + udotc);
        float Hi = W[l] * phi_norm * (CIX[l] * normx_val + CIY[l] * normy_val + CIZ[l] * normz_val);
        int offset = inline4D(ii,jj,kk,l,NX,NY,NZ);
        g[offset] = geq + Hi;
    }
}

// =================================================================================================== //



// =================================================================================================== //

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
    const float U_JET, const int DIAM,
    const int NX, const int NY, const int NZ,
    const int STEP, const int MACRO_SAVE
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (k != 0) return; 
    
    float cx = NX * 0.5f;
    float cy = NY * 0.5f;

    float dx = i - cx;
    float dy = j - cy;
    float Ri = sqrt(dx*dx + dy*dy);
    
    if (Ri > DIAM) return;

    float u_in = U_JET; //* (1.0f + DATAZ[STEP / MACRO_SAVE]);
    float phi_in = 0.5f + 0.5f * tanh(2.0f * (DIAM - Ri) / 3.0f);
    
    int idx_in = inline3D(i,j,k,NX,NY);

    float ffx_val = ffx[idx_in];
    float ffy_val = ffy[idx_in];
    float ffz_val = ffz[idx_in];

    rho[idx_in] = 1.0f;
    phi[idx_in] = phi_in;
    ux[idx_in] = 0.0f;
    uy[idx_in] = 0.0f;
    uz[idx_in] = u_in * phi_in; 

    float uz_val = uz[idx_in]; 
    float rho_val = rho[idx_in];

    float uu = 0.5f * (uz_val * uz_val) / CSSQ;
    float invRhoCssq = 1.0f / (rho_val * CSSQ);

    float auxHe = 1.0f - OMEGA / 2.0f;  

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        float udotc = (uz_val * CIZ[l]) / CSSQ;
        float feq = W[l] * (1.0f + (udotc + 0.5f * udotc*udotc - uu));

        float HeF = auxHe * feq * (CIX[l] * ffx_val +
                                   CIY[l] * ffy_val +
                                   (CIZ[l] - uz_val) * ffz_val) * invRhoCssq;
        
        int i_new = i + CIX[l];
        int j_new = j + CIY[l];
        int k_new = k + CIZ[l];

        if (i_new >= 0 && i_new < NX &&
            j_new >= 0 && j_new < NY &&
            k_new >= 0 && k_new < NZ) {
            int idx_f = inline4D(i_new,j_new,k_new,l,NX,NY,NZ);
            f[idx_f] = feq + HeF;
        }
    }

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        float udotc = (uz_val * CIZ[l]) / CSSQ;
        float geq = W[l] * phi[idx_in] * (1.0f + udotc);
        float Hi = W[l] * SHARP_C * phi[idx_in] * (1.0f - phi[idx_in]) *
                    (CIX[l] * normx[idx_in] + CIY[l] * normy[idx_in] + CIZ[l] * normz[idx_in]);

        int i_new = i + CIX[l];
        int j_new = j + CIY[l];
        int k_new = k + CIZ[l];

        if (i_new >= 0 && i_new < NX &&
            j_new >= 0 && j_new < NY &&
            k_new >= 0 && k_new < NZ) {
            int idx_g = inline4D(i_new,j_new,k_new,l,NX,NY,NZ);
            g[idx_g] = geq + Hi;
        }
    }
}

// =================================================================================================== //



// =================================================================================================== //
__global__ void normalizeUz(
    const float * __restrict__ uz,
    float * __restrict__ uz_norm,
    const float U_JET,
    const int NX, const int NY, const int NZ
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx = i + j * NX + k * NX * NY;
    uz_norm[idx] = uz[idx] / U_JET;
}
