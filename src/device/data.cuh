#pragma once
#include "common.cuh"
 
struct LBMFields {
    float *rho, *phi;
    float *ux, *uy, *uz;
    float *normx, *normy, *normz;
    float *ffx, *ffy, *ffz;
    float *f, *g;
};

extern LBMFields d;

void initializeVars();
