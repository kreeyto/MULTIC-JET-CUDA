#ifndef PRECISION_CUH
#define PRECISION_CUH

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

#define SINGLE_PRECISION
#define PRECISION_TYPE "float"

#define BLOCK_SIZE 4

#define IDX3D(i,j,k) ((i) + (j) * nx + (k) * nx * ny)
#define IDX4D(i,j,k,l) ((i) + (j) * nx + (k) * nx * ny + (l) * nx * ny * nz)
__device__ __forceinline__ int inline3D(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}
__device__ __forceinline__ int inline4D(int i, int j, int k, int l, int nx, int ny, int nz) {
    return inline3D(i,j,k,nx,ny) + l * nx * ny * nz;
}

#ifdef FD3Q19
    #define FPOINTS 19
#elif defined(FD3Q27)
    #define FPOINTS 27
#endif
#ifdef PD3Q19
    #define GPOINTS 19
#endif

#endif
