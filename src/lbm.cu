#include "kernels.cuh"

// ================================================================================================== //

__global__ void phiCalc(
    float * __restrict__ phi,
    const float * __restrict__ g,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float sum = 0.0f;       
    for (int l = 0; l < GPOINTS; ++l) {
        int idx4D = inline4D(i,j,k,l,nx,ny,nz);
        sum += g[idx4D];
    }

    phi[idx3D] = sum;
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gradCalc(
    const float * __restrict__ phi,
    float * __restrict__ normx,
    float * __restrict__ normy,
    float * __restrict__ normz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float grad_fix = 0.0f, grad_fiy = 0.0f, grad_fiz = 0.0f;
    for (int l = 0; l < FPOINTS; ++l) {
        int ii = i + CIX[l];
        int jj = j + CIY[l];
        int kk = k + CIZ[l];

        int offset = inline3D(ii,jj,kk,nx,ny);
        float val = phi[offset];
        float coef = 3.0f * W[l];
        grad_fix += coef * CIX[l] * val;
        grad_fiy += coef * CIY[l] * val;
        grad_fiz += coef * CIZ[l] * val;
    }

    float gmag_sq = grad_fix*grad_fix + grad_fiy*grad_fiy + grad_fiz*grad_fiz;
    float factor = rsqrtf(fmaxf(gmag_sq, 1e-9));

    normx[idx3D] = grad_fix * factor;
    normy[idx3D] = grad_fiy * factor;
    normz[idx3D] = grad_fiz * factor;
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void momentiCalc(
    float * __restrict__ ux,
    float * __restrict__ uy,
    float * __restrict__ uz,
    float * __restrict__ rho,
    const float * __restrict__ f,
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);
    
    float fneq[FPOINTS];
    float fVal[FPOINTS];

    for (int l = 0; l < FPOINTS; ++l) {
        int idx4D = inline4D(i,j,k,l,nx,ny,nz);
        fVal[l] = f[idx4D];
    }

    float rhoVal = 0.0f;
    for (int l = 0; l < FPOINTS; ++l)
        rhoVal += fVal[l];

    float invRho = 1.0f / rhoVal;

    float uxVal = invRho * (fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16]);
    float uyVal = invRho * (fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] + fVal[14] - fVal[13] + fVal[17] - fVal[18]);
    float uzVal = invRho * (fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] + fVal[16] - fVal[15] + fVal[18] - fVal[17]);

    float invCssq = 1.0f / CSSQ;
    float uu = 0.5f * (uxVal * uxVal + uyVal * uyVal + uzVal * uzVal) * invCssq;

    float sumXX = 0.0f, sumYY = 0.0f, sumZZ = 0.0f;
    float sumXY = 0.0f, sumXZ = 0.0f, sumYZ = 0.0f;

    for (int l = 0; l < FPOINTS; ++l) {
        float udotc = (uxVal * CIX[l] + uyVal * CIY[l] + uzVal * CIZ[l]) * invCssq;
        float udotc2 = udotc * udotc;
        float eqBase = rhoVal * (udotc + 0.5f * udotc2 - uu);
        float feq = W[l] * (rhoVal + eqBase);
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
    const float * __restrict__ rho,
    const float * __restrict__ pxx,
    const float * __restrict__ pyy,
    const float * __restrict__ pzz,
    const float * __restrict__ pxy,
    const float * __restrict__ pxz,
    const float * __restrict__ pyz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float ux_val = ux[idx3D], uy_val = uy[idx3D], uz_val = uz[idx3D], rho_val = rho[idx3D];
    float pxx_val = pxx[idx3D], pyy_val = pyy[idx3D], pzz_val = pzz[idx3D];
    float pxy_val = pxy[idx3D], pxz_val = pxz[idx3D], pyz_val = pyz[idx3D];

    float uu = 0.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val) / CSSQ;
    float invCssq = 1.0f / CSSQ;

    for (int l = 0; l < FPOINTS; ++l) {
        int ii = i + CIX[l]; 
        int jj = j + CIY[l]; 
        int kk = k + CIZ[l];
        
        float udotc = (ux_val * CIX[l] + uy_val * CIY[l] + uz_val * CIZ[l]) * invCssq;
        float feq = W[l] * (rho_val + rho_val * (udotc + 0.5f * udotc*udotc - uu));
        float fneq = (W[l] / (2.0f * CSSQ * CSSQ)) * ((CIX[l]*CIX[l] - CSSQ) * pxx_val +
                                                      (CIY[l]*CIY[l] - CSSQ) * pyy_val +
                                                      (CIZ[l]*CIZ[l] - CSSQ) * pzz_val +
                                                       2.0f * CIX[l] * CIY[l] * pxy_val +
                                                       2.0f * CIX[l] * CIZ[l] * pxz_val +
                                                       2.0f * CIY[l] * CIZ[l] * pyz_val
                                                    );
        int offset = inline4D(ii,jj,kk,l,nx,ny,nz);
        f[offset] = feq + (1.0f - OMEGA) * fneq; 
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
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float ux_val = ux[idx3D];
    float uy_val = uy[idx3D];
    float uz_val = uz[idx3D];
    float phi_val = phi[idx3D];
    float normx_val = normx[idx3D]; 
    float normy_val = normy[idx3D];
    float normz_val = normz[idx3D];

    float invCSSQ = 1.0f / CSSQ;

    float phi_norm = SHARP_C * phi_val * (1.0f - phi_val);
    for (int l = 0; l < GPOINTS; ++l) {
        int ii = i + CIX[l]; 
        int jj = j + CIY[l]; 
        int kk = k + CIZ[l];
        
        float udotc = (ux_val * CIX[l] + uy_val * CIY[l] + uz_val * CIZ[l]) * invCSSQ;
        float geq = W_G[l] * phi_val * (1.0f + udotc);
        float Hi = phi_norm * (CIX[l] * normx_val + CIY[l] * normy_val + CIZ[l] * normz_val);
        int offset = inline4D(ii,jj,kk,l,nx,ny,nz);
        g[offset] = geq + W_G[l] * Hi; // + (1 - omega) * gneq;
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
    float u_max, int d_half,
    int nx, int ny, int nz,
    int step, int MACRO_SAVE
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (k != 0) return;
    
    float nozzle_center_x = nx / 2.0f;
    float nozzle_center_y = ny / 2.0f;
    float nozzle_half = d_half;
    float u_in = u_max * (1 + DATAZ[step/MACRO_SAVE] * 1000);
    
    if (i < (nozzle_center_x - nozzle_half) || i > (nozzle_center_x + nozzle_half)) return;
    if (j < (nozzle_center_y - nozzle_half) || j > (nozzle_center_y + nozzle_half)) return;
    
    int inlet_k = 0.0f;
    int idx_inlet = inline3D(i, j, inlet_k, nx, ny);
    
    rho[idx_inlet] = 1.0f;
    phi[idx_inlet] = 1.0f;
    ux[idx_inlet] = 0.0f;
    uy[idx_inlet] = 0.0f;
    uz[idx_inlet] = u_in;
    
    float uu = 0.5f * (u_in * u_in) / CSSQ;
    
    for (int n = 0; n < FPOINTS; n++) {
        float udotc = (u_in * CIZ[n]) / CSSQ;
        float feq = W[n] * (1.0f + (udotc + 0.5f * udotc * udotc - uu));
        
        int i_new = i + CIX[n];
        int j_new = j + CIY[n];
        int k_new = inlet_k + CIZ[n];
        
        if (i_new >= 0 && i_new < nx &&
            j_new >= 0 && j_new < ny &&
            k_new >= 0 && k_new < nz) {
            int idx_f = inline4D(i_new, j_new, k_new, n, nx, ny, nz);
            f[idx_f] = feq;
        }
    }
    
    for (int n = 0; n < GPOINTS; n++) {
        float udotc = (u_in * CIZ[n]) / CSSQ;
        float feq_g = W_G[n] * phi[idx_inlet] * (1.0f + udotc);
        float Hi = SHARP_C * phi[idx_inlet] * (1.0f - phi[idx_inlet]) *
                    (CIX[n] * normx[idx_inlet] + CIY[n] * normy[idx_inlet] + CIZ[n] * normz[idx_inlet]);
        
        int idx_g_inlet = inline4D(i, j, inlet_k, n, nx, ny, nz);
        g[idx_g_inlet] = feq_g + W_G[n] * Hi;
        
        int i_new = i + CIX[n];
        int j_new = j + CIY[n];
        int k_new = inlet_k + CIZ[n];
        
        if (i_new >= 0 && i_new < nx &&
            j_new >= 0 && j_new < ny &&
            k_new >= 0 && k_new < nz) {
            int idx_g = inline4D(i_new, j_new, k_new, n, nx, ny, nz);
            g[idx_g] = g[idx_g_inlet];
        }
    }
}

// =================================================================================================== //

