#ifndef VAR_CUH
#define VAR_CUH

#include "header.cuh"

extern __constant__ float CSSQ, OMEGA, SHARP_C, INTERFACE_WIDTH, SIGMA;
extern __constant__ float W[NLINKS];
extern __constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];

#ifdef PERTURBATION
    extern __constant__ float DATAZ[200];
#endif
 
extern float *d_f, *d_g;
extern float *d_normx, *d_normy, *d_normz, *d_indicator;
extern float *d_ffx, *d_ffy, *d_ffz;
extern float *d_ux, *d_uy, *d_uz;
extern float *d_rho, *d_phi;

void initializeVars();

#endif
