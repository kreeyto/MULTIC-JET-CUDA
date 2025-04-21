#ifndef HEADER_CUH
#define HEADER_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <math.h>

using namespace std;

#define PRECISION_TYPE "float"

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8

#define IDX3D(x,y,z) ((x) + (y) * NX + (z) * NX * NY)
#define IDX4D(x,y,z,Q) ((x) + (y) * NX + (z) * NX * NY + (Q) * NX * NY * NZ)

__device__ __forceinline__ int idxGlobal3(int x, int y, int z, int NX, int NY) {
    return x + y * NX + z * NX * NY;
}
__device__ __forceinline__ int idxGlobal4(int x, int y, int z, int Q, int NX, int NY, int NZ) {
    int slice = NX * NY;
    return x + y * NX + z * slice + Q * slice * NZ;
}

__device__ __forceinline__ int idxShared3(int tx, int ty, int tz) {
    return tx + ty * BLOCK_X + tz * BLOCK_X * BLOCK_Y;
}

__device__ __forceinline__ int idxShared4(int tx, int ty, int tz, int Q) {
    int slice = BLOCK_X * BLOCK_Y;
    return tx + ty * BLOCK_X + tz * slice + Q * slice * BLOCK_Z;
}

#ifdef D3Q19
    constexpr int NLINKS = 19;
#elif defined(D3Q27)
    constexpr int NLINKS = 27;
#endif

#endif
