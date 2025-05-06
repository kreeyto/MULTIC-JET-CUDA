#include "kernels.cuh"
#include "deviceFunctions.cuh"

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

__global__ void gpuComputeInterface(LBMFields d) {
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

    const int idx3 = gpuIdxGlobal3(x,y,z);
    float grad_phi_x = 3.0f * (W[1]  * d.phi[gpuIdxGlobal3(x+1,y,z)]   - W[2]  * d.phi[gpuIdxGlobal3(x-1,y,z)]
                             + W[7]  * d.phi[gpuIdxGlobal3(x+1,y+1,z)] - W[8]  * d.phi[gpuIdxGlobal3(x-1,y-1,z)]
                             + W[9]  * d.phi[gpuIdxGlobal3(x+1,y,z+1)] - W[10] * d.phi[gpuIdxGlobal3(x-1,y,z-1)]
                             + W[13] * d.phi[gpuIdxGlobal3(x+1,y-1,z)] - W[14] * d.phi[gpuIdxGlobal3(x-1,y+1,z)]
                             + W[15] * d.phi[gpuIdxGlobal3(x+1,y,z-1)] - W[16] * d.phi[gpuIdxGlobal3(x-1,y,z+1)]);

    float grad_phi_y = 3.0f * (W[3]  * d.phi[gpuIdxGlobal3(x,y+1,z)]   - W[4]  * d.phi[gpuIdxGlobal3(x,y-1,z)]
                             + W[7]  * d.phi[gpuIdxGlobal3(x+1,y+1,z)] - W[8]  * d.phi[gpuIdxGlobal3(x-1,y-1,z)]
                             + W[11] * d.phi[gpuIdxGlobal3(x,y+1,z+1)] - W[12] * d.phi[gpuIdxGlobal3(x,y-1,z-1)]
                             - W[13] * d.phi[gpuIdxGlobal3(x+1,y-1,z)] + W[14] * d.phi[gpuIdxGlobal3(x-1,y+1,z)]
                             + W[17] * d.phi[gpuIdxGlobal3(x,y+1,z-1)] - W[18] * d.phi[gpuIdxGlobal3(x,y-1,z+1)]);

    float grad_phi_z = 3.0f * (W[5]  * d.phi[gpuIdxGlobal3(x,y,z+1)]   - W[6]  * d.phi[gpuIdxGlobal3(x,y,z-1)]
                             + W[9]  * d.phi[gpuIdxGlobal3(x+1,y,z+1)] - W[10] * d.phi[gpuIdxGlobal3(x-1,y,z-1)]
                             + W[11] * d.phi[gpuIdxGlobal3(x,y+1,z+1)] - W[12] * d.phi[gpuIdxGlobal3(x,y-1,z-1)]
                             - W[15] * d.phi[gpuIdxGlobal3(x+1,y,z-1)] + W[16] * d.phi[gpuIdxGlobal3(x-1,y,z+1)]
                             - W[17] * d.phi[gpuIdxGlobal3(x,y+1,z-1)] + W[18] * d.phi[gpuIdxGlobal3(x,y-1,z+1)]);

    float squared = grad_phi_x*grad_phi_x + grad_phi_y*grad_phi_y + grad_phi_z*grad_phi_z;
    float mag = rsqrtf(fmaxf(squared,1e-9));
    float normx_val = grad_phi_x * mag;
    float normy_val = grad_phi_y * mag;
    float normz_val = grad_phi_z * mag;
    float ind_val = squared * mag;

    d.normx[idx3] = normx_val;
    d.normy[idx3] = normy_val;
    d.normz[idx3] = normz_val;

    float curvature = -3.0f * (W[1]  *  d.normx[gpuIdxGlobal3(x+1,y,z)] - W[2] * d.normx[gpuIdxGlobal3(x-1,y,z)]
                             + W[3]  *  d.normy[gpuIdxGlobal3(x,y+1,z)] - W[4] * d.normy[gpuIdxGlobal3(x,y-1,z)]
                             + W[5]  *  d.normz[gpuIdxGlobal3(x,y,z+1)] - W[6] * d.normz[gpuIdxGlobal3(x,y,z-1)]
                             + W[7]  * (d.normx[gpuIdxGlobal3(x+1,y+1,z)] + d.normy[gpuIdxGlobal3(x+1,y+1,z)])
                             - W[8]  * (d.normx[gpuIdxGlobal3(x-1,y-1,z)] + d.normy[gpuIdxGlobal3(x-1,y-1,z)])
                             + W[9]  * (d.normx[gpuIdxGlobal3(x+1,y,z+1)] + d.normz[gpuIdxGlobal3(x+1,y,z+1)])
                             - W[10] * (d.normx[gpuIdxGlobal3(x-1,y,z+1)] + d.normz[gpuIdxGlobal3(x-1,y,z+1)])
                             + W[11] * (d.normy[gpuIdxGlobal3(x,y+1,z+1)] + d.normz[gpuIdxGlobal3(x,y+1,z+1)])
                             - W[12] * (d.normy[gpuIdxGlobal3(x,y-1,z+1)] + d.normz[gpuIdxGlobal3(x,y-1,z+1)])
                             + W[13] * (d.normx[gpuIdxGlobal3(x+1,y-1,z)] - d.normy[gpuIdxGlobal3(x+1,y-1,z)])
                             - W[14] * (d.normx[gpuIdxGlobal3(x-1,y+1,z)] - d.normy[gpuIdxGlobal3(x-1,y+1,z)])
                             + W[15] * (d.normx[gpuIdxGlobal3(x+1,y,z-1)] - d.normz[gpuIdxGlobal3(x+1,y,z-1)])
                             - W[16] * (d.normx[gpuIdxGlobal3(x-1,y,z-1)] - d.normz[gpuIdxGlobal3(x-1,y,z-1)])
                             + W[17] * (d.normy[gpuIdxGlobal3(x,y+1,z-1)] - d.normz[gpuIdxGlobal3(x,y+1,z-1)])
                             - W[18] * (d.normy[gpuIdxGlobal3(x,y-1,z+1)] - d.normz[gpuIdxGlobal3(x,y-1,z+1)]));   

    float coeff_curv = SIGMA * curvature;
    d.ffx[idx3] = coeff_curv * normx_val * ind_val;
    d.ffy[idx3] = coeff_curv * normy_val * ind_val;
    d.ffz[idx3] = coeff_curv * normz_val * ind_val;
}