#include "kernels.cuh"

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ - 1;
    if (x >= NX || y >= NY) return;

    const int idx3_out = gpuIdxGlobal3(x,y,z);
    const int idx3_prev = gpuIdxGlobal3(x,y,z-1);

    d.rho[idx3_out] = d.rho[idx3_prev];
    d.ux[idx3_out]  = d.ux[idx3_prev];
    d.uy[idx3_out]  = d.uy[idx3_prev];
    d.uz[idx3_out]  = d.uz[idx3_prev];
    d.phi[idx3_out] = d.phi[idx3_prev];

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4_out = gpuIdxGlobal4(x,y,z,Q);
        const int idx4_prev = gpuIdxGlobal4(x,y,z-1,Q);
        d.f[idx4_out] = d.f[idx4_prev];
        d.g[idx4_out] = d.g[idx4_prev];
    }
}

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;  

    if (x >= NX || y >= NY) return;

    float center_x = 0.5f * NX;
    float center_y = 0.5f * NY;

    float dx = x-center_x, dy = y-center_y;
    float radial_dist = sqrtf(dx*dx + dy*dy);
    
    float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    float radial_dist_norm = radial_dist / radius;

    //float phi_in = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
    float smoothing_factor = 1.0f - gpuSmoothstep(0.6f,1.0f,radial_dist_norm);
    float phi_in = smoothing_factor;

    #ifdef PERTURBATION
        float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 100) * phi_in; 
    #else
        float uz_in = U_JET * phi_in; 
    #endif
    
    const int idx3_in = gpuIdxGlobal3(x,y,z);

    float ffx_val = d.ffx[idx3_in];
    float ffy_val = d.ffy[idx3_in];
    float ffz_val = d.ffz[idx3_in];

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);

    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in; 

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpuComputeEquilibriaSecondOrder(rho_val,0.0f,0.0f,uz_in,uu,Q);
        float he_force = COEFF_HE * feq * (CIX[Q] * ffx_val +
                                           CIY[Q] * ffy_val +
                                          (CIZ[Q] - uz_in) * ffz_val) * 3.0f; 
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.f[streamed_idx4] = feq + he_force;
    }

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpuComputeEquilibriaFirstOrder(phi_in,0.0f,0.0f,uz_in,Q);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}

