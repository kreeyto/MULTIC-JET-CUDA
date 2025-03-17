#ifndef AUXFUNCTIONS_CUH
#define AUXFUNCTIONS_CUH

#include "header.cuh"

void freeMemory(float **pointers, int COUNT);

void generateSimulationInfoFile(
    const string& FILEPATH, int NX, int NY, int NZ, int STAMP, int NSTEPS, float TAU, 
    const string& SIM_ID, const string& VELOCITY_SET
);

void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const string& SIM_DIR, 
    const string& ID, int STEP, const string& VAR_NAME
);

#endif
