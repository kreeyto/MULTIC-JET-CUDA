#pragma once
#include "device_structs.cuh"
#include "device_functions.cuh"

__global__ void gpuInitDistributions(LBMFields d);

__global__ void gpuComputePhaseField(LBMFields d);
__global__ void gpuComputeGradients(LBMFields d);
__global__ void gpuComputeCurvature(LBMFields d);

__global__ void gpuFusedCollisionStream(LBMFields d);
__global__ void gpuApplyOutflow(LBMFields d);
__global__ void gpuEvolvePhaseField(LBMFields d);

__global__ void gpuApplyInflow(LBMFields d, const int STEP);
