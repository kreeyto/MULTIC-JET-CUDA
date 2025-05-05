#pragma once
#include "device/data.cuh"

__global__ void gpuInitDistributions(DeviceFields d);

// ============================================================================================== //

__global__ void gpuComputePhaseField(DeviceFields d);

__global__ void gpuComputeInterface(DeviceFields d);

__global__ void gpuFusedCollisionStream(DeviceFields d);

__global__ void gpuEvolveScalarField(DeviceFields d);

__global__ void gpuApplyInflowBoundary(DeviceFields d, const int STEP);