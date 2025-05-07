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
    
    float feq[NLINKS];
    float fneq[NLINKS];
    float pop[NLINKS];
    float he_term[NLINKS];
      
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
    #ifdef D3Q27
    pop[19] = d.f[gpuIdxGlobal4(x,y,z,19)];
    pop[20] = d.f[gpuIdxGlobal4(x,y,z,20)];
    pop[21] = d.f[gpuIdxGlobal4(x,y,z,21)];
    pop[22] = d.f[gpuIdxGlobal4(x,y,z,22)];
    pop[23] = d.f[gpuIdxGlobal4(x,y,z,23)];
    pop[24] = d.f[gpuIdxGlobal4(x,y,z,24)];
    pop[25] = d.f[gpuIdxGlobal4(x,y,z,25)];
    pop[26] = d.f[gpuIdxGlobal4(x,y,z,26)];
    #endif // D3Q27

    float rho_val = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    #ifdef D3Q27
    rho_val += pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
    #endif // D3Q27

    float inv_rho = 1.0f / rho_val;

    float mom_x = pop[1] - pop[2] + pop[7] - pop[8]  + pop[9]  - pop[10] + pop[13] - pop[14] + pop[15] - pop[16];
    float mom_y = pop[3] - pop[4] + pop[7] - pop[8]  + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18];
    float mom_z = pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17];
    #ifdef D3Q27
    mom_x += pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25];
    mom_y += pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]; 
    mom_z += pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26];
    #endif // D3Q27

    float sum_ux = inv_rho * mom_x;
    float sum_uy = inv_rho * mom_y;
    float sum_uz = inv_rho * mom_z;

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

    feq[0]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,0);
    feq[1]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,1);
    feq[2]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,2);
    feq[3]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,3);
    feq[4]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,4);
    feq[5]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,5);
    feq[6]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,6);
    feq[7]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,7);
    feq[8]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,8);
    feq[9]  = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,9);
    feq[10] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,10);
    feq[11] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,11);
    feq[12] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,12);
    feq[13] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,13);
    feq[14] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,14);
    feq[15] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,15);
    feq[16] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,16);
    feq[17] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,17);
    feq[18] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,18);
    #ifdef D3Q27
    feq[19] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,19);
    feq[20] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,20);
    feq[21] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,21);
    feq[22] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,22);
    feq[23] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,23);
    feq[24] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,24);
    feq[25] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,25);
    feq[26] = gpuComputeFeq(rho_val,ux_val,uy_val,uz_val,uu,26);
    #endif // D3Q27

    he_term[0]  = coeff_he * feq[0]  * (-ux_val*ffx_val       + -uy_val*ffy_val       + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[1]  = coeff_he * feq[1]  * ((1 - ux_val)*ffx_val  + -uy_val*ffy_val       + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[2]  = coeff_he * feq[2]  * ((-1 - ux_val)*ffx_val + -uy_val*ffy_val       + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[3]  = coeff_he * feq[3]  * (-ux_val*ffx_val       + (1 - uy_val)*ffy_val  + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[4]  = coeff_he * feq[4]  * (-ux_val*ffx_val       + (-1 - uy_val)*ffy_val + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[5]  = coeff_he * feq[5]  * (-ux_val*ffx_val       + -uy_val*ffy_val       + (1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[6]  = coeff_he * feq[6]  * (-ux_val*ffx_val       + -uy_val*ffy_val       + (-1 - uz_val)*ffz_val) * inv_rho_cssq;
    he_term[7]  = coeff_he * feq[7]  * ((1 - ux_val)*ffx_val  + (1 - uy_val)*ffy_val  + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[8]  = coeff_he * feq[8]  * ((-1 - ux_val)*ffx_val + (-1 - uy_val)*ffy_val + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[9]  = coeff_he * feq[9]  * ((1 - ux_val)*ffx_val  + -uy_val*ffy_val       + (1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[10] = coeff_he * feq[10] * ((-1 - ux_val)*ffx_val + -uy_val*ffy_val       + (-1 - uz_val)*ffz_val) * inv_rho_cssq;
    he_term[11] = coeff_he * feq[11] * (-ux_val*ffx_val       + (1 - uy_val)*ffy_val  + (1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[12] = coeff_he * feq[12] * (-ux_val*ffx_val       + (-1 - uy_val)*ffy_val + (-1 - uz_val)*ffz_val) * inv_rho_cssq;
    he_term[13] = coeff_he * feq[13] * ((1 - ux_val)*ffx_val  + (-1 - uy_val)*ffy_val + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[14] = coeff_he * feq[14] * ((-1 - ux_val)*ffx_val + (1 - uy_val)*ffy_val  + -uz_val*ffz_val)       * inv_rho_cssq;
    he_term[15] = coeff_he * feq[15] * ((1 - ux_val)*ffx_val  + -uy_val*ffy_val       + (-1 - uz_val)*ffz_val) * inv_rho_cssq;
    he_term[16] = coeff_he * feq[16] * ((-1 - ux_val)*ffx_val + -uy_val*ffy_val       + (1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[17] = coeff_he * feq[17] * (-ux_val*ffx_val       + (1 - uy_val)*ffy_val  + (-1 - uz_val)*ffz_val) * inv_rho_cssq;
    he_term[18] = coeff_he * feq[18] * (-ux_val*ffx_val       + (-1 - uy_val)*ffy_val + (1 - uz_val)*ffz_val)  * inv_rho_cssq;
    #ifdef D3Q27
    he_term[19] = coeff_he * feq[19] * ((1 - ux_val)*ffx_val  + (1 - uy_val)*ffy_val   + (1 - uz_val)*ffz_val)   * inv_rho_cssq;
    he_term[20] = coeff_he * feq[20] * ((-1 - ux_val)*ffx_val + (-1 - uy_val)*ffy_val  + (-1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[21] = coeff_he * feq[21] * ((1 - ux_val)*ffx_val  + (1 - uy_val)*ffy_val   + (-1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[22] = coeff_he * feq[22] * ((-1 - ux_val)*ffx_val + (-1 - uy_val)*ffy_val  + (1 - uz_val)*ffz_val)   * inv_rho_cssq;
    he_term[23] = coeff_he * feq[23] * ((1 - ux_val)*ffx_val  + (-1 - uy_val)*ffy_val  + (1 - uz_val)*ffz_val)   * inv_rho_cssq;
    he_term[24] = coeff_he * feq[24] * ((-1 - ux_val)*ffx_val + (1 - uy_val)*ffy_val   + (-1 - uz_val)*ffz_val)  * inv_rho_cssq;
    he_term[25] = coeff_he * feq[25] * ((-1 - ux_val)*ffx_val + (1 - uy_val)*ffy_val   + (1 - uz_val)*ffz_val)   * inv_rho_cssq;
    he_term[26] = coeff_he * feq[26] * ((1 - ux_val)*ffx_val  + (-1 - uy_val)*ffy_val  + (-1 - uz_val)*ffz_val)  * inv_rho_cssq;
    #endif // D3Q27

    fneq[0]  = pop[0]  - (feq[0]  - he_term[0] );
    fneq[1]  = pop[1]  - (feq[1]  - he_term[1] );
    fneq[2]  = pop[2]  - (feq[2]  - he_term[2] );
    fneq[3]  = pop[3]  - (feq[3]  - he_term[3] );
    fneq[4]  = pop[4]  - (feq[4]  - he_term[4] );
    fneq[5]  = pop[5]  - (feq[5]  - he_term[5] );
    fneq[6]  = pop[6]  - (feq[6]  - he_term[6] );
    fneq[7]  = pop[7]  - (feq[7]  - he_term[7] );
    fneq[8]  = pop[8]  - (feq[8]  - he_term[8] );
    fneq[9]  = pop[9]  - (feq[9]  - he_term[9] );
    fneq[10] = pop[10] - (feq[10] - he_term[10]);
    fneq[11] = pop[11] - (feq[11] - he_term[11]);
    fneq[12] = pop[12] - (feq[12] - he_term[12]);
    fneq[13] = pop[13] - (feq[13] - he_term[13]);
    fneq[14] = pop[14] - (feq[14] - he_term[14]);
    fneq[15] = pop[15] - (feq[15] - he_term[15]);
    fneq[16] = pop[16] - (feq[16] - he_term[16]);
    fneq[17] = pop[17] - (feq[17] - he_term[17]);
    fneq[18] = pop[18] - (feq[18] - he_term[18]);
    #ifdef D3Q27
    fneq[19] = pop[19] - (feq[19] - he_term[19]);
    fneq[20] = pop[20] - (feq[20] - he_term[20]);
    fneq[21] = pop[21] - (feq[21] - he_term[21]);
    fneq[22] = pop[22] - (feq[22] - he_term[22]);
    fneq[23] = pop[23] - (feq[23] - he_term[23]);
    fneq[24] = pop[24] - (feq[24] - he_term[24]);
    fneq[25] = pop[25] - (feq[25] - he_term[25]);
    fneq[26] = pop[26] - (feq[26] - he_term[26]);
    #endif // D3Q27

    float PXX = fneq[1]  + fneq[2]  + fneq[7]  + fneq[8]  + fneq[9]  + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
    float PYY = fneq[3]  + fneq[4]  + fneq[7]  + fneq[8]  + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
    float PZZ = fneq[5]  + fneq[6]  + fneq[9]  + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
    float PXY = fneq[7]  - fneq[13] + fneq[8]  - fneq[14];
    float PXZ = fneq[9]  - fneq[15] + fneq[10] - fneq[16];
    float PYZ = fneq[11] - fneq[17] + fneq[12] - fneq[18];
    #ifdef D3Q27
    PXX += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PYY += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PZZ += fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23] + fneq[24] + fneq[25] + fneq[26];
    PXY += fneq[19] - fneq[23] + fneq[20] - fneq[24] + fneq[21] - fneq[25] + fneq[22] - fneq[26];
    PXZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[23] - fneq[25] + fneq[24] - fneq[26];
    PYZ += fneq[19] - fneq[21] + fneq[20] - fneq[22] + fneq[25] - fneq[23] + fneq[26] - fneq[24];
    #endif // D3Q27

    fneq[0] =  (W[0]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,0);
    fneq[1] =  (W[1]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,1);
    fneq[2] =  (W[2]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,2);
    fneq[3] =  (W[3]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,3);
    fneq[4] =  (W[4]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,4);
    fneq[5] =  (W[5]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,5);
    fneq[6] =  (W[6]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,6);
    fneq[7] =  (W[7]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,7);
    fneq[8] =  (W[8]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,8);
    fneq[9] =  (W[9]  * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,9);
    fneq[10] = (W[10] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,10);
    fneq[11] = (W[11] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,11);
    fneq[12] = (W[12] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,12);
    fneq[13] = (W[13] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,13);
    fneq[14] = (W[14] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,14);
    fneq[15] = (W[15] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,15);
    fneq[16] = (W[16] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,16);
    fneq[17] = (W[17] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,17);
    fneq[18] = (W[18] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,18);
    #ifdef D3Q27
    fneq[19] = (W[19] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,19);
    fneq[20] = (W[20] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,20);
    fneq[21] = (W[21] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,21);
    fneq[22] = (W[22] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,22);
    fneq[23] = (W[23] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,23);
    fneq[24] = (W[24] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,24);
    fneq[25] = (W[25] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,25);
    fneq[26] = (W[26] * 4.5f) * gpuComputeTensor2(PXX,PYY,PZZ,PXY,PXZ,PYZ,26);
    #endif // D3Q27

    float omc = 1.0f - OMEGA;

    d.f[gpuIdxGlobal4(x,y,z,0)]      = feq[0]  + omc * fneq[0]  + he_term[0];
    d.f[gpuIdxGlobal4(x+1,y,z,1)]    = feq[1]  + omc * fneq[1]  + he_term[1];
    d.f[gpuIdxGlobal4(x-1,y,z,2)]    = feq[2]  + omc * fneq[2]  + he_term[2];
    d.f[gpuIdxGlobal4(x,y+1,z,3)]    = feq[3]  + omc * fneq[3]  + he_term[3];
    d.f[gpuIdxGlobal4(x,y-1,z,4)]    = feq[4]  + omc * fneq[4]  + he_term[4];
    d.f[gpuIdxGlobal4(x,y,z+1,5)]    = feq[5]  + omc * fneq[5]  + he_term[5];
    d.f[gpuIdxGlobal4(x,y,z-1,6)]    = feq[6]  + omc * fneq[6]  + he_term[6];
    d.f[gpuIdxGlobal4(x+1,y+1,z,7)]  = feq[7]  + omc * fneq[7]  + he_term[7];
    d.f[gpuIdxGlobal4(x-1,y-1,z,8)]  = feq[8]  + omc * fneq[8]  + he_term[8];
    d.f[gpuIdxGlobal4(x+1,y,z+1,9)]  = feq[9]  + omc * fneq[9]  + he_term[9];
    d.f[gpuIdxGlobal4(x-1,y,z-1,10)] = feq[10] + omc * fneq[10] + he_term[10];
    d.f[gpuIdxGlobal4(x,y+1,z+1,11)] = feq[11] + omc * fneq[11] + he_term[11];
    d.f[gpuIdxGlobal4(x,y-1,z-1,12)] = feq[12] + omc * fneq[12] + he_term[12];
    d.f[gpuIdxGlobal4(x+1,y-1,z,13)] = feq[13] + omc * fneq[13] + he_term[13];
    d.f[gpuIdxGlobal4(x-1,y+1,z,14)] = feq[14] + omc * fneq[14] + he_term[14];
    d.f[gpuIdxGlobal4(x+1,y,z-1,15)] = feq[15] + omc * fneq[15] + he_term[15];
    d.f[gpuIdxGlobal4(x-1,y,z+1,16)] = feq[16] + omc * fneq[16] + he_term[16];
    d.f[gpuIdxGlobal4(x,y+1,z-1,17)] = feq[17] + omc * fneq[17] + he_term[17];
    d.f[gpuIdxGlobal4(x,y-1,z+1,18)] = feq[18] + omc * fneq[18] + he_term[18];
    #ifdef D3Q27
    d.f[gpuIdxGlobal4(x+1,y+1,z+1,19)] = feq[19] + omc * fneq[19] + he_term[19];
    d.f[gpuIdxGlobal4(x-1,y-1,z-1,20)] = feq[20] + omc * fneq[20] + he_term[20];
    d.f[gpuIdxGlobal4(x+1,y+1,z-1,21)] = feq[21] + omc * fneq[21] + he_term[21];
    d.f[gpuIdxGlobal4(x-1,y-1,z+1,22)] = feq[22] + omc * fneq[22] + he_term[22];
    d.f[gpuIdxGlobal4(x+1,y-1,z+1,23)] = feq[23] + omc * fneq[23] + he_term[23];
    d.f[gpuIdxGlobal4(x-1,y+1,z-1,24)] = feq[24] + omc * fneq[24] + he_term[24];
    d.f[gpuIdxGlobal4(x-1,y+1,z+1,25)] = feq[25] + omc * fneq[25] + he_term[25];
    d.f[gpuIdxGlobal4(x+1,y-1,z-1,26)] = feq[26] + omc * fneq[26] + he_term[26];
    #endif // D3Q27
 
    d.ux[idx3] = ux_val; d.uy[idx3] = uy_val; d.uz[idx3] = uz_val;
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

    float geq[NLINKS];
    float anti_diff[NLINKS];

    float ux_val = d.ux[idx3];
    float uy_val = d.uy[idx3];
    float uz_val = d.uz[idx3];
    float phi_val = d.phi[idx3];
    float normx_val = d.normx[idx3]; 
    float normy_val = d.normy[idx3];
    float normz_val = d.normz[idx3];
    
    float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);
    float phi_norm = SHARP_C * phi_val * (1.0f - phi_val);

    geq[0]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,0);
    geq[1]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,1);
    geq[2]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,2);
    geq[3]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,3);
    geq[4]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,4);
    geq[5]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,5);
    geq[6]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,6);
    geq[7]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,7);
    geq[8]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,8);
    geq[9]  = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,9);
    geq[10] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,10);
    geq[11] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,11);
    geq[12] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,12);
    geq[13] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,13);
    geq[14] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,14);
    geq[15] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,15);
    geq[16] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,16);
    geq[17] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,17);
    geq[18] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,18);
    #ifdef D3Q27
    geq[19] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,19);
    geq[20] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,20);
    geq[21] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,21);
    geq[22] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,22);
    geq[23] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,23);
    geq[24] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,24);
    geq[25] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,25);
    geq[26] = gpuComputeFeq(phi_val,ux_val,uy_val,uz_val,uu,26);
    #endif

    anti_diff[0]  = 0;
    anti_diff[1]  = W[1]  * phi_norm * normx_val;
    anti_diff[2]  = W[2]  * phi_norm * (-1 * normx_val);
    anti_diff[3]  = W[3]  * phi_norm * normy_val;
    anti_diff[4]  = W[4]  * phi_norm * (-1 * normy_val);
    anti_diff[5]  = W[5]  * phi_norm * normz_val;
    anti_diff[6]  = W[6]  * phi_norm * (-1 * normz_val);
    anti_diff[7]  = W[7]  * phi_norm * (normx_val + normy_val);
    anti_diff[8]  = W[8]  * phi_norm * (-1 * normx_val + -1 * normy_val);
    anti_diff[9]  = W[9]  * phi_norm * (normx_val + normz_val);
    anti_diff[10] = W[10] * phi_norm * (-1 * normx_val + -1 * normz_val);
    anti_diff[11] = W[11] * phi_norm * (normy_val + normz_val);
    anti_diff[12] = W[12] * phi_norm * (-1 * normy_val + -1 * normz_val);
    anti_diff[13] = W[13] * phi_norm * (normx_val + -1 * normy_val);
    anti_diff[14] = W[14] * phi_norm * (-1 * normx_val + normy_val);
    anti_diff[15] = W[15] * phi_norm * (normx_val + -1 * normz_val);
    anti_diff[16] = W[16] * phi_norm * (-1 * normx_val + normz_val);
    anti_diff[17] = W[17] * phi_norm * (normy_val + -1 * normz_val);
    anti_diff[18] = W[18] * phi_norm * (-1 * normy_val + normz_val);
    #ifdef D3Q27
    anti_diff[19] = W[19] * phi_norm * (normx_val + normy_val + normz_val);
    anti_diff[20] = W[20] * phi_norm * (-1 * normx_val + -1 * normy_val + -1 * normz_val);
    anti_diff[21] = W[21] * phi_norm * (normx_val + normy_val + -1 * normz_val);
    anti_diff[22] = W[22] * phi_norm * (-1 * normx_val + -1 * normy_val + normz_val);
    anti_diff[23] = W[23] * phi_norm * (normx_val + -1 * normy_val + normz_val);
    anti_diff[24] = W[24] * phi_norm * (-1 * normx_val + normy_val + -1 * normz_val);
    anti_diff[25] = W[25] * phi_norm * (-1 * normx_val + normy_val + normz_val);
    anti_diff[26] = W[26] * phi_norm * (normx_val + -1 * normy_val + -1 * normz_val);
    #endif

    d.g[gpuIdxGlobal4(x,y,z,0)]      = geq[0]; // zeroth order term cancels out
    d.g[gpuIdxGlobal4(x+1,y,z,1)]    = geq[1]  + anti_diff[1];
    d.g[gpuIdxGlobal4(x-1,y,z,2)]    = geq[2]  + anti_diff[2];
    d.g[gpuIdxGlobal4(x,y+1,z,3)]    = geq[3]  + anti_diff[3];
    d.g[gpuIdxGlobal4(x,y-1,z,4)]    = geq[4]  + anti_diff[4];
    d.g[gpuIdxGlobal4(x,y,z+1,5)]    = geq[5]  + anti_diff[5];
    d.g[gpuIdxGlobal4(x,y,z-1,6)]    = geq[6]  + anti_diff[6];
    d.g[gpuIdxGlobal4(x+1,y+1,z,7)]  = geq[7]  + anti_diff[7];
    d.g[gpuIdxGlobal4(x-1,y-1,z,8)]  = geq[8]  + anti_diff[8];
    d.g[gpuIdxGlobal4(x+1,y,z+1,9)]  = geq[9]  + anti_diff[9];
    d.g[gpuIdxGlobal4(x-1,y,z-1,10)] = geq[10] + anti_diff[10];
    d.g[gpuIdxGlobal4(x,y+1,z+1,11)] = geq[11] + anti_diff[11];
    d.g[gpuIdxGlobal4(x,y-1,z-1,12)] = geq[12] + anti_diff[12];
    d.g[gpuIdxGlobal4(x+1,y-1,z,13)] = geq[13] + anti_diff[13];
    d.g[gpuIdxGlobal4(x-1,y+1,z,14)] = geq[14] + anti_diff[14];
    d.g[gpuIdxGlobal4(x+1,y,z-1,15)] = geq[15] + anti_diff[15];
    d.g[gpuIdxGlobal4(x-1,y,z+1,16)] = geq[16] + anti_diff[16];
    d.g[gpuIdxGlobal4(x,y+1,z-1,17)] = geq[17] + anti_diff[17];
    d.g[gpuIdxGlobal4(x,y-1,z+1,18)] = geq[18] + anti_diff[18];
    #ifdef D3Q27
    d.g[gpuIdxGlobal4(x+1,y+1,z+1,19)] = geq[19] + anti_diff[19];
    d.g[gpuIdxGlobal4(x-1,y-1,z-1,20)] = geq[20] + anti_diff[20];
    d.g[gpuIdxGlobal4(x+1,y+1,z-1,21)] = geq[21] + anti_diff[21];
    d.g[gpuIdxGlobal4(x-1,y-1,z+1,22)] = geq[22] + anti_diff[22];
    d.g[gpuIdxGlobal4(x+1,y-1,z+1,23)] = geq[23] + anti_diff[23];
    d.g[gpuIdxGlobal4(x-1,y+1,z-1,24)] = geq[24] + anti_diff[24];
    d.g[gpuIdxGlobal4(x-1,y+1,z+1,25)] = geq[25] + anti_diff[25];
    d.g[gpuIdxGlobal4(x+1,y-1,z-1,26)] = geq[26] + anti_diff[26];
    #endif
    
}

