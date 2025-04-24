#ifndef AUXFUNCTIONS_CUH
#define AUXFUNCTIONS_CUH

#include "header.cuh"

void freeMemory(float **pointers, int COUNT);

void generateSimulationInfoFile(
    const string& FILEPATH, const int MACRO_SAVE, const int NSTEPS, const float TAU, 
    const string& SIM_ID, const string& VELOCITY_SET
);

void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const string& SIM_DIR, 
    const string& ID, int STEP, const string& VAR_NAME
);

#endif
