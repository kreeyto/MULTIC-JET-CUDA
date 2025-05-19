#include "kernels.cuh"

__global__ void gpuComputePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float phi_val = d.g[gpuIdxGlobal4(x,y,z,0)] + d.g[gpuIdxGlobal4(x,y,z,1)] + d.g[gpuIdxGlobal4(x,y,z,2)] 
                  + d.g[gpuIdxGlobal4(x,y,z,3)] + d.g[gpuIdxGlobal4(x,y,z,4)] + d.g[gpuIdxGlobal4(x,y,z,5)] 
                  + d.g[gpuIdxGlobal4(x,y,z,6)];

    d.phi[idx3] = phi_val;
}

__global__ void gpuComputeGradients(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    /*
    float grad_phi_x = 0.0f, grad_phi_y = 0.0f, grad_phi_z = 0.0f;
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX_G[Q];
        const int yy = y + CIY_G[Q];
        const int zz = z + CIZ_G[Q];
        const int propagated_idx3 = gpuIdxGlobal3(xx,yy,zz);

        float phi_prop = d.phi[propagated_idx3];
        float coeff = 3.0f * W_G[Q];
        grad_phi_x += coeff * CIX_G[Q] * phi_prop;
        grad_phi_y += coeff * CIY_G[Q] * phi_prop;
        grad_phi_z += coeff * CIZ_G[Q] * phi_prop;
    }
    */

    float grad_phi_x = 3.0f * (W_G[1] * d.phi[gpuIdxGlobal3(x+1,y,z)] - W_G[2] * d.phi[gpuIdxGlobal3(x-1,y,z)]);
    float grad_phi_y = 3.0f * (W_G[3] * d.phi[gpuIdxGlobal3(x,y+1,z)] - W_G[4] * d.phi[gpuIdxGlobal3(x,y-1,z)]);
    float grad_phi_z = 3.0f * (W_G[5] * d.phi[gpuIdxGlobal3(x,y,z+1)] - W_G[6] * d.phi[gpuIdxGlobal3(x,y,z-1)]);

    float squared = grad_phi_x*grad_phi_x + grad_phi_y*grad_phi_y + grad_phi_z*grad_phi_z;
    float mag = rsqrtf(fmaxf(squared,1e-9f));
    float normx_val = grad_phi_x * mag;
    float normy_val = grad_phi_y * mag;
    float normz_val = grad_phi_z * mag;
    float ind_val = squared * mag;

    d.normx[idx3] = normx_val;
    d.normy[idx3] = normy_val;
    d.normz[idx3] = normz_val;
    d.ind[idx3] = ind_val;
}

__global__ void gpuComputeCurvature(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ ||
        x == 0 || x == NX-1 ||
        y == 0 || y == NY-1 ||
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float normx_val = d.normx[idx3];
    float normy_val = d.normy[idx3];
    float normz_val = d.normz[idx3];
    float ind_val = d.ind[idx3];

    /*
    float curvature = 0.0f;
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX_G[Q];
        const int yy = y + CIY_G[Q];
        const int zz = z + CIZ_G[Q];
        const int propagated_idx3 = gpuIdxGlobal3(xx,yy,zz);
        
        float normx_prop = d.normx[propagated_idx3];
        float normy_prop = d.normy[propagated_idx3];
        float normz_prop = d.normz[propagated_idx3];
        float coeff_curv = 3.0f * W_G[Q];
        curvature -= coeff_curv * (CIX_G[Q]*normx_prop + CIY_G[Q]*normy_prop + CIZ_G[Q]*normz_prop);
    }
    */

    float curvature = -3.0f * ( W_G[1] * d.normx[gpuIdxGlobal3(x+1,y,z)] 
                              - W_G[2] * d.normx[gpuIdxGlobal3(x-1,y,z)] 
                              + W_G[3] * d.normy[gpuIdxGlobal3(x,y+1,z)] 
                              - W_G[4] * d.normy[gpuIdxGlobal3(x,y-1,z)]
                              + W_G[5] * d.normz[gpuIdxGlobal3(x,y,z+1)]
                              - W_G[6] * d.normz[gpuIdxGlobal3(x,y,z-1)] );

    float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx_val * ind_val;
    d.ffy[idx3] = coeff_force * normy_val * ind_val;
    d.ffz[idx3] = coeff_force * normz_val * ind_val;
}