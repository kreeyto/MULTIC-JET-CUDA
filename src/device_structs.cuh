#pragma once
#include "common.cuh"
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *normx, *normy, *normz, *ind;
    float *ffx, *ffy, *ffz;
    float *f, *g; 
};

extern LBMFields lbm;

void initDeviceVars();
