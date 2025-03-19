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

#define IDX3D(i,j,k) ((i) + (j) * NX + (k) * NX * NY)
#define IDX4D(i,j,k,l) ((i) + (j) * NX + (k) * NX * NY + (l) * NX * NY * NZ)
__device__ __forceinline__ int inline3D(int i, int j, int k, int NX, int NY) {
    return i + j * NX + k * NX * NY;
}
__device__ __forceinline__ int inline4D(int i, int j, int k, int l, int NX, int NY, int NZ) {
    return inline3D(i,j,k,NX,NY) + l * NX * NY * NZ;
}

#ifdef D3Q19
    #define NLINKS 19
#elif defined(D3Q27)
    #define NLINKS 27
#endif

#endif
