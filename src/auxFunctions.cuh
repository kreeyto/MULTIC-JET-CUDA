#ifndef AUXFUNCTIONS_CUH
#define AUXFUNCTIONS_CUH

#include "header.cuh"

void freeMemory(float **pointers, int count);

void generateSimulationInfoFile(
    const string& filepath, int nx, int ny, int nz, int stamp, int nsteps, float tau, 
    const string& sim_id, const string& fluid_model
);

void copyAndSaveToBinary(
    const float* d_data, size_t size, const string& sim_dir, 
    const string& id, int t, const string& var_name
);

#endif
