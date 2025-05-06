#include "kernels.cuh"
#include "deviceFunctions.cuh"

__global__ void gpuFusedCollisionStream(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const int idx3 = gpuIdxGlobal3(x,y,z);
    
    float fneq[NLINKS];
    float pop[NLINKS];
      
    pop[0]  = d.f[gpuIdxGlobal4(x,y,z,0)];
    pop[1]  = d.f[gpuIdxGlobal4(x,y,z,1)];
    pop[2]  = d.f[gpuIdxGlobal4(x,y,z,2)];
    pop[3]  = d.f[gpuIdxGlobal4(x,y,z,3)];
    pop[4]  = d.f[gpuIdxGlobal4(x,y,z,4)];
    pop[5]  = d.f[gpuIdxGlobal4(x,y,z,5)];
    pop[6]  = d.f[gpuIdxGlobal4(x,y,z,6)];
    pop[7]  = d.f[gpuIdxGlobal4(x,y,z,7)];
    pop[8]  = d.f[gpuIdxGlobal4(x,y,z,8)];
    pop[9]  = d.f[gpuIdxGlobal4(x,y,z,9)];
    pop[10] = d.f[gpuIdxGlobal4(x,y,z,10)];
    pop[11] = d.f[gpuIdxGlobal4(x,y,z,11)];
    pop[12] = d.f[gpuIdxGlobal4(x,y,z,12)];
    pop[13] = d.f[gpuIdxGlobal4(x,y,z,13)];
    pop[14] = d.f[gpuIdxGlobal4(x,y,z,14)];
    pop[15] = d.f[gpuIdxGlobal4(x,y,z,15)];
    pop[16] = d.f[gpuIdxGlobal4(x,y,z,16)];
    pop[17] = d.f[gpuIdxGlobal4(x,y,z,17)];
    pop[18] = d.f[gpuIdxGlobal4(x,y,z,18)];

    float rho_val = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    float inv_rho = 1.0f / rho_val;

    float sum_ux = inv_rho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
    float sum_uy = inv_rho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
    float sum_uz = inv_rho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);

    float ffx_val = d.ffx[idx3];
    float ffy_val = d.ffy[idx3];
    float ffz_val = d.ffz[idx3];

    float fx_corr = ffx_val * 0.5f * inv_rho;
    float fy_corr = ffy_val * 0.5f * inv_rho;
    float fz_corr = ffz_val * 0.5f * inv_rho;

    float ux_val = sum_ux + fx_corr;
    float uy_val = sum_uy + fy_corr;
    float uz_val = sum_uz + fz_corr;

    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);
    float inv_rho_cssq = 3.0f * inv_rho;
    float coeff_he = 1.0f - OMEGA / 2.0f;

    fneq[0]  = pop[0] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,0) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,0) * ((CIX[0] - ux_val) * ffx_val + (CIY[0] - uy_val) * ffy_val + (CIZ[0] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[1]  = pop[1] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,1) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,1) * ((CIX[1] - ux_val) * ffx_val + (CIY[1] - uy_val) * ffy_val + (CIZ[1] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[2]  = pop[2] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,2) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,2) * ((CIX[2] - ux_val) * ffx_val + (CIY[2] - uy_val) * ffy_val + (CIZ[2] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[3]  = pop[3] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,3) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,3) * ((CIX[3] - ux_val) * ffx_val + (CIY[3] - uy_val) * ffy_val + (CIZ[3] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[4]  = pop[4] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,4) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,4) * ((CIX[4] - ux_val) * ffx_val + (CIY[4] - uy_val) * ffy_val + (CIZ[4] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[5]  = pop[5] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,5) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,5) * ((CIX[5] - ux_val) * ffx_val + (CIY[5] - uy_val) * ffy_val + (CIZ[5] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[6]  = pop[6] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,6) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,6) * ((CIX[6] - ux_val) * ffx_val + (CIY[6] - uy_val) * ffy_val + (CIZ[6] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[7]  = pop[7] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,7) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,7) * ((CIX[7] - ux_val) * ffx_val + (CIY[7] - uy_val) * ffy_val + (CIZ[7] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[8]  = pop[8] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,8) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,8) * ((CIX[8] - ux_val) * ffx_val + (CIY[8] - uy_val) * ffy_val + (CIZ[8] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[9]  = pop[9] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,9) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,9) * ((CIX[9] - ux_val) * ffx_val + (CIY[9] - uy_val) * ffy_val + (CIZ[9] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[10] = pop[10] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,10) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,10) * ((CIX[10] - ux_val) * ffx_val + (CIY[10] - uy_val) * ffy_val + (CIZ[10] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[11] = pop[11] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,11) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,11) * ((CIX[11] - ux_val) * ffx_val + (CIY[11] - uy_val) * ffy_val + (CIZ[11] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[12] = pop[12] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,12) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,12) * ((CIX[12] - ux_val) * ffx_val + (CIY[12] - uy_val) * ffy_val + (CIZ[12] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[13] = pop[13] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,13) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,13) * ((CIX[13] - ux_val) * ffx_val + (CIY[13] - uy_val) * ffy_val + (CIZ[13] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[14] = pop[14] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,14) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,14) * ((CIX[14] - ux_val) * ffx_val + (CIY[14] - uy_val) * ffy_val + (CIZ[14] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[15] = pop[15] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,15) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,15) * ((CIX[15] - ux_val) * ffx_val + (CIY[15] - uy_val) * ffy_val + (CIZ[15] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[16] = pop[16] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,16) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,16) * ((CIX[16] - ux_val) * ffx_val + (CIY[16] - uy_val) * ffy_val + (CIZ[16] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[17] = pop[17] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,17) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,17) * ((CIX[17] - ux_val) * ffx_val + (CIY[17] - uy_val) * ffy_val + (CIZ[17] - uz_val) * ffz_val) * inv_rho_cssq);
    fneq[18] = pop[18] - (gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,18) - coeff_he * gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,18) * ((CIX[18] - ux_val) * ffx_val + (CIY[18] - uy_val) * ffy_val + (CIZ[18] - uz_val) * ffz_val) * inv_rho_cssq);

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

__global__ void gpuEvolveScalarField(LBMFields d) {
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

