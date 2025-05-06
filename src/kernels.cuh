#pragma once
#include "deviceStructs.cuh"

__global__ void gpuInitDistributions(LBMFields d);

__global__ void gpuComputePhaseField(LBMFields d);

__global__ void gpuComputeInterface(LBMFields d);

__global__ void gpuFusedCollisionStream(LBMFields d);

__global__ void gpuEvolveScalarField(LBMFields d);

__global__ void gpuApplyInflowBoundary(LBMFields d, const int STEP);