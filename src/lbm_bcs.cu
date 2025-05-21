#include "kernels.cuh"

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    bool isValidEdge = (x < NX && y < NY) && (x == 0 || x == NX-1 || y == 0 || y == NY-1);
    if (!isValidEdge) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int dz = z + CIZ[Q];
        if (dz >= 0 && dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            //if (xx >= 0 && xx < NX && yy >= 0 && yy < NY) {
                const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
                d.f[streamed_idx4] = d.rho[idx3] * W[Q];
            //}
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int dz = z + CIZ[Q];
        if (dz >= 0 && dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            //if (xx >= 0 && xx < NX && yy >= 0 && yy < NY) {
                const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
                d.g[streamed_idx4] = d.phi[idx3] * W_G[Q];
            //}
        }
    }
}

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    float center_x = (NX - 1) * 0.5f;
    float center_y = (NY - 1) * 0.5f;

    float dx = x-center_x, dy = y-center_y;
    float radial_dist = sqrtf(dx*dx + dy*dy);

    float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    float radial_dist_norm = radial_dist / radius;
    float envelope = 1.0f - gpuSmoothstep(0.6f, 1.0f, radial_dist_norm);
    float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
    float phi_in = envelope * profile;

    #ifdef PERTURBATION
        float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
    #else
        float uz_in = U_JET * phi_in;
    #endif

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);

    int idx3_in = gpuIdxGlobal3(x,y,z);
    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int dz = CIZ[Q];
        if (z + dz >= 0 && z + dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            float feq = gpuComputeEquilibriaSecondOrder(rho_val,0.0f,0.0f,uz_in,uu,Q);
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
            d.f[streamed_idx4] = feq;
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int dz = CIZ[Q];
        if (z + dz >= 0 && z + dz < NZ) {
            const int xx = x + CIX[Q];
            const int yy = y + CIY[Q];
            float geq = gpuComputeEquilibriaFirstOrder(phi_in,0.0f,0.0f,uz_in,Q);
            const int streamed_idx4 = gpuIdxGlobal4(xx,yy,dz,Q);
            d.g[streamed_idx4] = geq;
        }
    }
}

__global__ void gpuApplyOutflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z != NZ - 1) return;

    const int idx_top = gpuIdxGlobal3(x, y, NZ - 1);
    const int idx_below = gpuIdxGlobal3(x, y, NZ - 2);

    d.phi[idx_top] = d.phi[idx_below];
}



