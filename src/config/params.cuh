#pragma once
using namespace std;

//#define RUN_MODE
#define SAMPLE_MODE
//#define PERTURBATION  

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

#ifdef D3Q19
    constexpr int NLINKS = 19;
#elif defined(D3Q27)
    constexpr int NLINKS = 27;
#endif

extern __constant__ float CSSQ;
extern __constant__ float OMEGA;
extern __constant__ float SHARP_C;
extern __constant__ float INTERFACE_WIDTH;
extern __constant__ float SIGMA;
extern __constant__ float W[NLINKS];
extern __constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];

#ifdef PERTURBATION
    extern __constant__ float DATAZ[200];
#endif

constexpr int MESH = 64;
constexpr int DIAM = (MESH + 9) / 10;
constexpr int NX = MESH;
constexpr int NY = MESH;
constexpr int NZ = MESH*2;

constexpr float U_JET = 0.05f;

constexpr int REYNOLDS = 5000;
constexpr int WEBER = 500;

constexpr float VISC = (U_JET * DIAM) / REYNOLDS;
constexpr float H_TAU = 0.5f + 3.0f * VISC;
constexpr float H_CSSQ = 1.0f / 3.0f;
constexpr float H_OMEGA = 1.0f / H_TAU;
constexpr float H_INTERFACE_WIDTH = 7.0f;
constexpr float H_SHARP_C = 0.15f * H_INTERFACE_WIDTH;
constexpr float H_SIGMA = (U_JET * U_JET * DIAM) / WEBER;

#ifdef RUN_MODE
    constexpr int MACRO_SAVE = 100, NSTEPS = 25000;
#elif defined(SAMPLE_MODE)
    constexpr int MACRO_SAVE = 100, NSTEPS = 1000;
#else
    constexpr int MACRO_SAVE = 1, NSTEPS = 0;
#endif

