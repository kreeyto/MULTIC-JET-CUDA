#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef D3Q19
    constexpr int NLINKS = 19;
#elif defined(D3Q27)
    constexpr int NLINKS = 27;
#endif

constexpr int MESH = 64;
constexpr int DIAM = (MESH + 9) / 10;
constexpr int NX = MESH;
constexpr int NY = MESH;
constexpr int NZ = MESH*4;

constexpr float U_JET = 0.05f;

#endif
