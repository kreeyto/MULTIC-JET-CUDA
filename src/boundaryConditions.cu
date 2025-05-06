#include "kernels.cuh"
#include "deviceFunctions.cuh"

__global__ void gpuApplyInflowBoundary(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;  

    if (x >= NX || y >= NY) return;

    float center_x = NX * 0.5f;
    float center_y = NY * 0.5f;

    float dx = x-center_x, dy = y-center_y;
    float geometry = sqrtf(dx*dx + dy*dy);
    
    if (geometry > DIAM) return;
    float geometry_norm = geometry / DIAM;

    float smoothing_factor = 1.0f - gpuSmoothstep(0.6f,1.0f,geometry_norm);
    float phi_in = smoothing_factor;

    #ifdef PERTURBATION
        float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10) * phi_in; 
    #else
        float uz_in = U_JET * phi_in; 
    #endif
    
    const int idx3_in = gpuIdxGlobal3(x,y,z);

    float ffx_val = d.ffx[idx3_in];
    float ffy_val = d.ffy[idx3_in];
    float ffz_val = d.ffz[idx3_in];

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);
    float coeff_he = 1.0f - OMEGA / 2.0f;  

    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in; 

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float feq = gpuComputeFeq(1.0f,0.0f,0.0f,uz_in,uu,Q);
        float he_force = coeff_he * feq * (CIX[Q] * ffx_val +
                                           CIY[Q] * ffy_val +
                                          (CIZ[Q] - uz_in) * ffz_val) * 3.0f; 
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.f[streamed_idx4] = feq + he_force;
    }

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float geq = gpuComputeFeq(phi_in,0.0f,0.0f,uz_in,uu,Q);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}