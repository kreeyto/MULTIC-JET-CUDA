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
#include "constants.cuh"

using namespace std;

#define PRECISION_TYPE "float"

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

#define IDX3D(x,y,z) ((x) + (y) * NX + (z) * NX * NY)
#define IDX4D(x,y,z,Q) ((x) + (y) * NX + (z) * NX * NY + (Q) * NX * NY * NZ)

#endif
