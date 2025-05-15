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

    float phi_val = 0.0f;
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        phi_val += d.g[idx4];
    }

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

    float grad_phi_x = 0.0f, grad_phi_y = 0.0f, grad_phi_z = 0.0f;
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        const int propagated_idx3 = gpuIdxGlobal3(xx,yy,zz);

        float phi_prop = d.phi[propagated_idx3];
        float coeff = 3.0f * W[Q];
        grad_phi_x += coeff * CIX[Q] * phi_prop;
        grad_phi_y += coeff * CIY[Q] * phi_prop;
        grad_phi_z += coeff * CIZ[Q] * phi_prop;
    }

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

    float curvature = 0.0f;
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        const int propagated_idx3 = gpuIdxGlobal3(xx,yy,zz);
        
        float normx_prop = d.normx[propagated_idx3];
        float normy_prop = d.normy[propagated_idx3];
        float normz_prop = d.normz[propagated_idx3];
        float coeff_curv = 3.0f * W[Q];
        curvature -= coeff_curv * (CIX[Q]*normx_prop + CIY[Q]*normy_prop + CIZ[Q]*normz_prop);
    }

    float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx_val * ind_val;
    d.ffy[idx3] = coeff_force * normy_val * ind_val;
    d.ffz[idx3] = coeff_force * normz_val * ind_val;
}