#include "var.cuh"

int mesh = 64;
int diam = 10; 

// extern vars
int nx = mesh; int ny = mesh; int nz = mesh*4;  
int d_half = diam/2;
float u_max = 0.1; 

__constant__ float TAU;
__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float SHARP_C;
__constant__ float SIGMA;
__constant__ float W[FPOINTS], W_G[GPOINTS];
__constant__ int CIX[FPOINTS], CIY[FPOINTS], CIZ[FPOINTS];

float *d_f, *d_g;
float *d_normx, *d_normy, *d_normz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

// ========================================================================== parametros ========================================================================== //

// ========== adimensional parameters ========== //
        //const float REYNOLDS = 100;
        //const float WEBER = 200;
// ============================================= //

//const float VISC = U_MAX + (DIAM+DIAM) / REYNOLDS;
//const float H_TAU = 0.5 + 3.0 * VISC;
const float h_tau = 0.505;
const float h_cssq = 1.0 / 3.0;
const float h_omega = 1.0 / h_tau;
const float h_sharp_c = 0.15 * 3.0;
//const float H_SIGMA = U_MAX * U_MAX * (DIAM+DIAM) / WEBER;
const float h_sigma = 0.1;

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

}

