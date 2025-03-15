#include "var.cuh"

int mesh = 64;
int diam = 10; 

// extern vars
int nx = mesh; int ny = mesh; int nz = mesh*4;  
int d_half = diam/2;
float u_max = 0.03; 

__constant__ float TAU;
__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float SHARP_C;
__constant__ float SIGMA;
__constant__ float W[FPOINTS], W_G[GPOINTS];
__constant__ int CIX[FPOINTS], CIY[FPOINTS], CIZ[FPOINTS];
__constant__ float DATAZ[200];

float *d_f, *d_g;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

// ========================================================================== parametros ========================================================================== //

// ========== adimensional parameters ========== //
        //const float REYNOLDS = 5000;
        const float WEBER = 1000;
// ============================================= //

//const float visc = u_max + (diam+diam) / REYNOLDS;
//const float h_tau = 0.5 + 3.0 * visc;
const float h_tau = 0.505;
const float h_cssq = 1.0 / 3.0;
const float h_omega = 1.0 / h_tau;
const float h_sharp_c = 0.15 * 3.0;
//const float h_sigma = 0.1;
const float h_sigma = u_max * u_max * (diam+diam) / WEBER;

// fluid velocity set
#ifdef FD3Q19
    int h_cix[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
    int h_ciy[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1 };
    int h_ciz[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1 };
#elif defined(FD3Q27)
    int h_cix[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    int h_ciy[27] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    int h_ciz[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
#endif

// fluid weights
#ifdef FD3Q19
    float h_w[19] = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
#elif defined(FD3Q27)
    float h_w[27] = {
        8.0 / 27.0,
        2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 
        1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0
    };
#endif

// phase field weights
#ifdef PD3Q19
    float h_w_g[19] = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
#endif

// perturbation
float h_dataz[200] = { 0.00079383, 0.00081679, 0.00002621, -0.00002419, -0.00044200, -0.00084266, 0.00048380, 0.00021733, 0.00032251, 0.00001137, 
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
    size_t size = nx * ny * nz * sizeof(float);            
    size_t f_size = nx * ny * nz * FPOINTS * sizeof(float); 
    size_t g_size = nx * ny * nz * GPOINTS * sizeof(float); 

    cudaMalloc((void **)&d_rho, size);
    cudaMalloc((void **)&d_phi, size);
    cudaMalloc((void **)&d_ux, size);
    cudaMalloc((void **)&d_uy, size);
    cudaMalloc((void **)&d_uz, size);
    cudaMalloc((void **)&d_normx, size);
    cudaMalloc((void **)&d_normy, size);
    cudaMalloc((void **)&d_normz, size);
    cudaMalloc((void **)&d_curvature, size);
    cudaMalloc((void **)&d_indicator, size);
    cudaMalloc((void **)&d_ffx, size);
    cudaMalloc((void **)&d_ffy, size);
    cudaMalloc((void **)&d_ffz, size);
    cudaMalloc((void **)&d_pxx, size);
    cudaMalloc((void **)&d_pyy, size);
    cudaMalloc((void **)&d_pzz, size);
    cudaMalloc((void **)&d_pxy, size);
    cudaMalloc((void **)&d_pxz, size);
    cudaMalloc((void **)&d_pyz, size);

    cudaMalloc((void **)&d_f, f_size);
    cudaMalloc((void **)&d_g, g_size);

    cudaMemset(d_phi, 0, size);
    cudaMemset(d_ux, 0, size);
    cudaMemset(d_uy, 0, size);
    cudaMemset(d_uz, 0, size);
    
    cudaMemset(d_f, 0, f_size);
    cudaMemset(d_g, 0, g_size);

    cudaMemset(d_normx, 0, size);
    cudaMemset(d_normy, 0, size);
    cudaMemset(d_normz, 0, size);
    cudaMemset(d_curvature, 0, size);
    cudaMemset(d_indicator, 0, size);
    cudaMemset(d_ffx, 0, size);
    cudaMemset(d_ffy, 0, size);
    cudaMemset(d_ffz, 0, size);

    cudaMemcpyToSymbol(TAU, &h_tau, sizeof(float));
    cudaMemcpyToSymbol(CSSQ, &h_cssq, sizeof(float));
    cudaMemcpyToSymbol(OMEGA, &h_omega, sizeof(float));
    cudaMemcpyToSymbol(SHARP_C, &h_sharp_c, sizeof(float));
    cudaMemcpyToSymbol(SIGMA, &h_sigma, sizeof(float));

    cudaMemcpyToSymbol(W, &h_w, FPOINTS * sizeof(float));
    cudaMemcpyToSymbol(W_G, &h_w_g, GPOINTS * sizeof(float));

    cudaMemcpyToSymbol(CIX, &h_cix, FPOINTS * sizeof(int));
    cudaMemcpyToSymbol(CIY, &h_ciy, FPOINTS * sizeof(int));
    cudaMemcpyToSymbol(CIZ, &h_ciz, FPOINTS * sizeof(int));

    cudaMemcpyToSymbol(DATAZ, &h_dataz, 200 * sizeof(float));

}

