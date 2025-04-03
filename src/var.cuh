#pragma once

#include "header.cuh"

extern int DIAM;
extern int NX, NY, NZ;
extern float U_JET;

extern __constant__ float CSSQ, OMEGA, SHARP_C, SIGMA;
extern __constant__ float W[NLINKS];
extern __constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];
extern __constant__ float DATAZ[200];
 
extern float *d_f, *d_g;
extern float *d_normx, *d_normy, *d_normz, *d_indicator;
extern float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
extern float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
extern float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

void initializeVars();
