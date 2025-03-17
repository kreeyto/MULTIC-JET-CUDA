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

#define PRECISION_TYPE "float"

#define IDX3D(i,j,k) ((i) + (j) * nx + (k) * nx * ny)
#define IDX4D(i,j,k,l) ((i) + (j) * nx + (k) * nx * ny + (l) * nx * ny * nz)
__device__ __forceinline__ int inline3D(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}
__device__ __forceinline__ int inline4D(int i, int j, int k, int l, int nx, int ny, int nz) {
    return inline3D(i,j,k,nx,ny) + l * nx * ny * nz;
}

#ifdef D3Q19
    #define NLINKS 19
#elif defined(D3Q27)
    #define NLINKS 27
#endif

#endif
