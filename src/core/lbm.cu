#include "core/kernels.cuh"
#include "device/functions.cuh"

__global__ void gpuInitDistributions(DeviceFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        d.f[idx4] = W[Q];
    }
}

// =================================================================================================== //

__global__ void gpuComputePhaseField(DeviceFields d) {
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
        phi_val += __ldg(&d.g[idx4]);
    }

    d.phi[idx3] = phi_val;
}

__global__ void gpuComputeInterface(DeviceFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

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

    float curvature = 0.0f;
    for (int Q = 0; Q < NLINKS; ++Q) {
        int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        int streamed_idx3 = gpuIdxGlobal3(xx, yy, zz);
        float coeff = 3.0f * W[Q];
        curvature -= coeff * (CIX[Q] * d.normx[streamed_idx3] +
                              CIY[Q] * d.normy[streamed_idx3] +
                              CIZ[Q] * d.normz[streamed_idx3]);
    }

    float coeff_curv = SIGMA * curvature;
    d.ffx[idx3] = coeff_curv * normx_val * ind_val;
    d.ffy[idx3] = coeff_curv * normy_val * ind_val;
    d.ffz[idx3] = coeff_curv * normz_val * ind_val;
}


// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuFusedCollisionStream(DeviceFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);
    
    float fneq[NLINKS];
    float f_val[NLINKS];
      
    // if casting f to shared memory i should load it here to kill NLINKS registers 
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int idx4 = gpuIdxGlobal4(x,y,z,Q);
        f_val[Q] = d.f[idx4];
    }

    float rho_val = f_val[0] + f_val[1] + f_val[2] + f_val[3] + f_val[4] + f_val[5] +
                    f_val[6] + f_val[7] + f_val[8] + f_val[9] + f_val[10] + f_val[11] +
                    f_val[12] + f_val[13] + f_val[14] + f_val[15] + f_val[16] + f_val[17] + f_val[18];

    float inv_rho = 1.0f / rho_val;

    float sum_ux = inv_rho * (f_val[1] - f_val[2] + f_val[7] - f_val[8] + f_val[9] - f_val[10] + f_val[13] - f_val[14] + f_val[15] - f_val[16]);
    float sum_uy = inv_rho * (f_val[3] - f_val[4] + f_val[7] - f_val[8] + f_val[11] - f_val[12] + f_val[14] - f_val[13] + f_val[17] - f_val[18]);
    float sum_uz = inv_rho * (f_val[5] - f_val[6] + f_val[9] - f_val[10] + f_val[11] - f_val[12] + f_val[16] - f_val[15] + f_val[18] - f_val[17]);

    float ffx_val = d.ffx[idx3];
    float ffy_val = d.ffy[idx3];
    float ffz_val = d.ffz[idx3];

    float half_ffx = ffx_val * 0.5f * inv_rho;
    float half_ffy = ffy_val * 0.5f * inv_rho;
    float half_ffz = ffz_val * 0.5f * inv_rho;

    float ux_val = sum_ux + half_ffx;
    float uy_val = sum_uy + half_ffy;
    float uz_val = sum_uz + half_ffz;

    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);
    float inv_rho_cssq = 3.0f * inv_rho;

    float coeff_he = 1.0f - OMEGA / 2.0f;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        float pre_feq = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,Q);
        float he_force = coeff_he * pre_feq * ((CIX[Q] - ux_val) * ffx_val +
                                               (CIY[Q] - uy_val) * ffy_val +
                                               (CIZ[Q] - uz_val) * ffz_val) * inv_rho_cssq;
        float feq = pre_feq - he_force; 
        fneq[Q] = f_val[Q] - feq;
    }

    float PXX = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7] - fneq[13] + fneq[8] - fneq[14];
    float PXZ = fneq[9] - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];

    d.ux[idx3] = ux_val; d.uy[idx3] = uy_val; d.uz[idx3] = uz_val;

    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float feq = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,Q);
        float he_force = coeff_he * feq * ( (CIX[Q] - ux_val) * ffx_val +
                                            (CIY[Q] - uy_val) * ffy_val +
                                            (CIZ[Q] - uz_val) * ffz_val ) * inv_rho_cssq;
        float fneq = (W[Q] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,Q);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.f[streamed_idx4] = feq + (1.0f - OMEGA) * fneq + he_force; 
    }
}

__global__ void gpuEvolveScalarField(DeviceFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);

    float ux_val = d.ux[idx3];
    float uy_val = d.uy[idx3];
    float uz_val = d.uz[idx3];
    float phi_val = d.phi[idx3];
    float normx_val = d.normx[idx3]; 
    float normy_val = d.normy[idx3];
    float normz_val = d.normz[idx3];
    
    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);
    float phi_norm = SHARP_C * phi_val * (1.0f - phi_val);
    #pragma unroll NLINKS
    for (int Q = 0; Q < NLINKS; ++Q) {
        const int xx = x + CIX[Q], yy = y + CIY[Q], zz = z + CIZ[Q];
        float geq = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,Q);
        float hi = W[Q] * phi_norm * (CIX[Q] * normx_val + CIY[Q] * normy_val + CIZ[Q] * normz_val);
        const int streamed_idx4 = gpuIdxGlobal4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq + hi;
    }
}

// =================================================================================================== //



// =================================================================================================== //

__global__ void gpuApplyInflowBoundary(DeviceFields d, const int STEP) {
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

// =================================================================================================== //

