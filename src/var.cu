#include "var.cuh"

int MESH = 64;
int DIAM = 8; // will be integer if MESH = 2^x

// extern vars
int NX = MESH; int NY = MESH; int NZ = MESH*4;  
int D_HALF = round(DIAM / 2);
float U_MAX = 0.1f; 

__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float SHARP_C;
__constant__ float SIGMA;
__constant__ float W[NLINKS];
__constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];
__constant__ float DATAZ[200];

float *d_f, *d_g;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

// ========================================================================== parametros ========================================================================== //

// ========== adimensional parameters ========== //
        //const int REYNOLDS = 5000;
        const int WEBER = 1000;
// ============================================= //

//const float VISC = U_MAX + (DIAM+DIAM) / REYNOLDS;
//const float H_TAU = 0.5f + 3.0f * VISC;
const float H_TAU = 0.505f;
const float H_CSSQ = 1.0f / 3.0f;
const float H_OMEGA = 1.0f / H_TAU;
const float H_SHARP_C = 0.15f * 3.0f;
const float H_SIGMA = U_MAX * U_MAX * (DIAM+DIAM) / WEBER;

// velocity set
#ifdef D3Q19
    int H_CIX[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
    int H_CIY[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1 };
    int H_CIZ[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1 };
#elif defined(D3Q27)
    int H_CIX[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    int H_CIY[27] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    int H_CIZ[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
#endif

// vs weights
#ifdef D3Q19
    float H_W[19] = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
#elif defined(D3Q27)
    float H_W[27] = {
        8.0 / 27.0,
        2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 
        1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0
    };
#endif

// perturbation
float H_DATAZ[200] = { 0.00079383, 0.00081679, 0.00002621, -0.00002419, -0.00044200, -0.00084266, 0.00048380, 0.00021733, 0.00032251, 0.00001137, 
                      -0.00050303, -0.00008389, 0.00000994, -0.00061235, 0.00092132, 0.00001801, 0.00064784, -0.00013657, 0.00051558, 0.00020564, 
                      -0.00074830, -0.00094143, -0.00052143, 0.00073746, 0.00024430, 0.00036541, -0.00014634, -0.00034321, 0.00013730, 0.00005668, 
                       0.00034116, -0.00098297, 0.00007028, 0.00042728, -0.00086542, -0.00059119, 0.00059534, 0.00026490, -0.00007748, -0.00054852, 
                      -0.00039547, 0.00009244, -0.00016603, 0.00003809, 0.00057867, 0.00036876, -0.00098247, -0.00071294, 0.00099262, 0.00018596, 
                      -0.00025951, -0.00067508, -0.00034442, 0.00004329, 0.00052225, -0.00026905, 0.00067835, 0.00072271, -0.00019486, -0.00097031, 
                       0.00080641, -0.00095198, -0.00007856, -0.00012953, 0.00044508, -0.00021542, -0.00016924, 0.00049395, 0.00059422, -0.00006069, 
                       0.00069688, 0.00031164, -0.00086361, 0.00051087, 0.00075494, -0.00058256, 0.00067235, 0.00070165, 0.00088299, 0.00085143, 
                      -0.00040871, -0.00000741, -0.00085449, -0.00075362, -0.00080573, 0.00020063, -0.00001421, -0.00093398, 0.00022559, 0.00074277, 
                      -0.00094501, 0.00096696, 0.00003558, -0.00049148, 0.00054682, -0.00066242, -0.00069007, -0.00026005, 0.00020265, 0.00091499, 
                      -0.00054173, 0.00025756, -0.00057015, -0.00063640, -0.00040327, -0.00092048, -0.00057386, -0.00018224, -0.00060635, 0.00033103, 
                       0.00023893, -0.00029650, -0.00053987, 0.00067523, 0.00067282, -0.00031058, -0.00079529, 0.00044863, 0.00085339, 0.00025606, 
                       0.00005468, -0.00086148, 0.00079563, 0.00048100, -0.00013505, 0.00021489, -0.00069042, 0.00039699, 0.00080755, -0.00082483, 
                       0.00047788, -0.00071238, 0.00018310, -0.00021486, 0.00088903, -0.00093828, -0.00045933, 0.00017546, 0.00097415, 0.00035564, 
                       0.00029083, -0.00094149, 0.00049215, -0.00070605, 0.00064217, -0.00046830, -0.00028556, -0.00019632, -0.00028125, 0.00098444, 
                      -0.00078697, 0.00063941, -0.00016519, 0.00019510, 0.00026044, -0.00037241, -0.00045767, 0.00025914, 0.00002784, 0.00021836, 
                       0.00021581, 0.00074161, 0.00051495, 0.00059711, -0.00084965, 0.00025144, -0.00067714, 0.00053914, 0.00018297, 0.00090897, 
                       0.00011948, -0.00092672, -0.00064307, -0.00032715, -0.00040575, -0.00044485, 0.00028828, -0.00099615, -0.00017845, 0.00052521, 
                      -0.00045545, 0.00011635, 0.00093167, 0.00062180, -0.00010542, 0.00085383, -0.00048304, -0.00042307, 0.00085464, 0.00005302, 
                      -0.00070889, 0.00045034, 0.00002412, -0.00016850, 0.00014029, 0.00036591, -0.00049267, 0.00049268, -0.00012600, -0.00017574 };
// =============================================================================================================================================================== //

void initializeVars() {
    size_t SIZE = NX * NY * NZ * sizeof(float);            
    size_t F_SIZE = NX * NY * NZ * NLINKS * sizeof(float); 
    size_t G_SIZE = NX * NY * NZ * NLINKS * sizeof(float); 

    cudaMalloc((void **)&d_rho, SIZE);
    cudaMalloc((void **)&d_phi, SIZE);
    cudaMalloc((void **)&d_ux, SIZE);
    cudaMalloc((void **)&d_uy, SIZE);
    cudaMalloc((void **)&d_uz, SIZE);
    cudaMalloc((void **)&d_normx, SIZE);
    cudaMalloc((void **)&d_normy, SIZE);
    cudaMalloc((void **)&d_normz, SIZE);
    cudaMalloc((void **)&d_curvature, SIZE);
    cudaMalloc((void **)&d_indicator, SIZE);
    cudaMalloc((void **)&d_ffx, SIZE);
    cudaMalloc((void **)&d_ffy, SIZE);
    cudaMalloc((void **)&d_ffz, SIZE);
    cudaMalloc((void **)&d_pxx, SIZE);
    cudaMalloc((void **)&d_pyy, SIZE);
    cudaMalloc((void **)&d_pzz, SIZE);
    cudaMalloc((void **)&d_pxy, SIZE);
    cudaMalloc((void **)&d_pxz, SIZE);
    cudaMalloc((void **)&d_pyz, SIZE);

    cudaMalloc((void **)&d_f, F_SIZE);
    cudaMalloc((void **)&d_g, G_SIZE);

    cudaMemset(d_phi, 0, SIZE);
    cudaMemset(d_ux, 0, SIZE);
    cudaMemset(d_uy, 0, SIZE);
    cudaMemset(d_uz, 0, SIZE);
    
    cudaMemset(d_f, 0, F_SIZE);
    cudaMemset(d_g, 0, G_SIZE);

    cudaMemset(d_normx, 0, SIZE);
    cudaMemset(d_normy, 0, SIZE);
    cudaMemset(d_normz, 0, SIZE);
    cudaMemset(d_curvature, 0, SIZE);
    cudaMemset(d_indicator, 0, SIZE);
    cudaMemset(d_ffx, 0, SIZE);
    cudaMemset(d_ffy, 0, SIZE);
    cudaMemset(d_ffz, 0, SIZE);

    cudaMemcpyToSymbol(CSSQ, &H_CSSQ, sizeof(float));
    cudaMemcpyToSymbol(OMEGA, &H_OMEGA, sizeof(float));
    cudaMemcpyToSymbol(SHARP_C, &H_SHARP_C, sizeof(float));
    cudaMemcpyToSymbol(SIGMA, &H_SIGMA, sizeof(float));

    cudaMemcpyToSymbol(W, &H_W, NLINKS * sizeof(float));

    cudaMemcpyToSymbol(CIX, &H_CIX, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIY, &H_CIY, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIZ, &H_CIZ, NLINKS * sizeof(int));

    cudaMemcpyToSymbol(DATAZ, &H_DATAZ, 200 * sizeof(float));

}

