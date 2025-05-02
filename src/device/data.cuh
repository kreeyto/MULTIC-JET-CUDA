#pragma once
#include "common.cuh"
 
extern float *d_f, *d_g;
extern float *d_normx, *d_normy, *d_normz;
extern float *d_ffx, *d_ffy, *d_ffz;
extern float *d_ux, *d_uy, *d_uz;
extern float *d_rho, *d_phi;

void initializeVars();
