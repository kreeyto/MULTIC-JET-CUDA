#include "var.cuh"

int MESH = 64;

// extern vars
int DIAM = ceil(MESH/10);
int NX = MESH; int NY = MESH; int NZ = MESH*4;  
float U_JET = 0.05f; 

__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float SHARP_C;
__constant__ float SIGMA;
__constant__ float W[NLINKS];
__constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];
//__constant__ float DATAZ[200];

float *d_f, *d_g;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

// ========================================================================== parametros ========================================================================== //

// ========== adimensional parameters ========== //
        const int REYNOLDS = 5000;
        const int WEBER = 500;
// ============================================= //

const float VISC = (U_JET * DIAM) / REYNOLDS;
const float H_TAU = 0.5f + 3.0f * VISC;
const float H_CSSQ = 1.0f / 3.0f;
const float H_OMEGA = 1.0f / H_TAU;
const float H_SHARP_C = 0.15f * 3.0f;
const float H_SIGMA = (U_JET * U_JET * DIAM) / WEBER;

// velocity set
#ifdef D3Q19
    int H_CIX[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
    int H_CIY[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1 };
    int H_CIZ[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1 };
#elif defined(D3Q27)
    int H_CIX[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    int H_CIY[27] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    int H_CIZ[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
#endif

// vs weights
#ifdef D3Q19
    float H_W[19] = {
        1.0f / 3.0f, 
        1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };
#elif defined(D3Q27)
    float H_W[27] = {
        8.0f / 27.0f,
        2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
        1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
        1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f
    };
#endif

// perturbation
/*float H_DATAZ[200] = { 0.00079383f, 0.00081679f, 0.00002621f, -0.00002419f, -0.00044200f, -0.00084266f, 0.00048380f, 0.00021733f, 0.00032251f, 0.00001137f, 
                      -0.00050303f, -0.00008389f, 0.00000994f, -0.00061235f, 0.00092132f, 0.00001801f, 0.00064784f, -0.00013657f, 0.00051558f, 0.00020564f, 
                      -0.00074830f, -0.00094143f, -0.00052143f, 0.00073746f, 0.00024430f, 0.00036541f, -0.00014634f, -0.00034321f, 0.00013730f, 0.00005668f, 
                       0.00034116f, -0.00098297f, 0.00007028f, 0.00042728f, -0.00086542f, -0.00059119f, 0.00059534f, 0.00026490f, -0.00007748f, -0.00054852f, 
                      -0.00039547f, 0.00009244f, -0.00016603f, 0.00003809f, 0.00057867f, 0.00036876f, -0.00098247f, -0.00071294f, 0.00099262f, 0.00018596f, 
                      -0.00025951f, -0.00067508f, -0.00034442f, 0.00004329f, 0.00052225f, -0.00026905f, 0.00067835f, 0.00072271f, -0.00019486f, -0.00097031f, 
                       0.00080641f, -0.00095198f, -0.00007856f, -0.00012953f, 0.00044508f, -0.00021542f, -0.00016924f, 0.00049395f, 0.00059422f, -0.00006069f, 
                       0.00069688f, 0.00031164f, -0.00086361f, 0.00051087f, 0.00075494f, -0.00058256f, 0.00067235f, 0.00070165f, 0.00088299f, 0.00085143f, 
                      -0.00040871f, -0.00000741f, -0.00085449f, -0.00075362f, -0.00080573f, 0.00020063f, -0.00001421f, -0.00093398f, 0.00022559f, 0.00074277f, 
                      -0.00094501f, 0.00096696f, 0.00003558f, -0.00049148f, 0.00054682f, -0.00066242f, -0.00069007f, -0.00026005f, 0.00020265f, 0.00091499f, 
                      -0.00054173f, 0.00025756f, -0.00057015f, -0.00063640f, -0.00040327f, -0.00092048f, -0.00057386f, -0.00018224f, -0.00060635f, 0.00033103f, 
                       0.00023893f, -0.00029650f, -0.00053987f, 0.00067523f, 0.00067282f, -0.00031058f, -0.00079529f, 0.00044863f, 0.00085339f, 0.00025606f, 
                       0.00005468f, -0.00086148f, 0.00079563f, 0.00048100f, -0.00013505f, 0.00021489f, -0.00069042f, 0.00039699f, 0.00080755f, -0.00082483f, 
                       0.00047788f, -0.00071238f, 0.00018310f, -0.00021486f, 0.00088903f, -0.00093828f, -0.00045933f, 0.00017546f, 0.00097415f, 0.00035564f, 
                       0.00029083f, -0.00094149f, 0.00049215f, -0.00070605f, 0.00064217f, -0.00046830f, -0.00028556f, -0.00019632f, -0.00028125f, 0.00098444f, 
                      -0.00078697f, 0.00063941f, -0.00016519f, 0.00019510f, 0.00026044f, -0.00037241f, -0.00045767f, 0.00025914f, 0.00002784f, 0.00021836f, 
                       0.00021581f, 0.00074161f, 0.00051495f, 0.00059711f, -0.00084965f, 0.00025144f, -0.00067714f, 0.00053914f, 0.00018297f, 0.00090897f, 
                       0.00011948f, -0.00092672f, -0.00064307f, -0.00032715f, -0.00040575f, -0.00044485f, 0.00028828f, -0.00099615f, -0.00017845f, 0.00052521f, 
                      -0.00045545f, 0.00011635f, 0.00093167f, 0.00062180f, -0.00010542f, 0.00085383f, -0.00048304f, -0.00042307f, 0.00085464f, 0.00005302f, 
                      -0.00070889f, 0.00045034f, 0.00002412f, -0.00016850f, 0.00014029f, 0.00036591f, -0.00049267f, 0.00049268f, -0.00012600f, -0.00017574f };*/
                                         
// =============================================================================================================================================================== //

void initializeVars() {
    size_t SIZE = NX * NY * NZ * sizeof(float);            
    size_t DIST_SIZE = NX * NY * NZ * NLINKS * sizeof(float); 

    cudaMalloc((void **)&d_rho, SIZE);
    cudaMalloc((void **)&d_phi, SIZE);
    cudaMalloc((void **)&d_ux, SIZE);
    cudaMalloc((void **)&d_uy, SIZE);
    cudaMalloc((void **)&d_uz, SIZE);
    cudaMalloc((void **)&d_normx, SIZE);
    cudaMalloc((void **)&d_normy, SIZE);
    cudaMalloc((void **)&d_normz, SIZE);
    cudaMalloc((void **)&d_indicator, SIZE);
    cudaMalloc((void **)&d_ffx, SIZE);
    cudaMalloc((void **)&d_ffy, SIZE);
    cudaMalloc((void **)&d_ffz, SIZE);
    cudaMalloc((void **)&d_pxx, SIZE);
    cudaMalloc((void **)&d_pyy, SIZE);
    cudaMalloc((void **)&d_pzz, SIZE);
    cudaMalloc((void **)&d_pxy, SIZE);
    cudaMalloc((void **)&d_pxz, SIZE);
    cudaMalloc((void **)&d_pyz, SIZE);

    cudaMalloc((void **)&d_f, DIST_SIZE);
    cudaMalloc((void **)&d_g, DIST_SIZE );

    cudaMemset(d_phi, 0, SIZE);
    cudaMemset(d_ux, 0, SIZE);
    cudaMemset(d_uy, 0, SIZE);
    cudaMemset(d_uz, 0, SIZE);
    
    cudaMemset(d_f, 0, DIST_SIZE );
    cudaMemset(d_g, 0, DIST_SIZE );

    cudaMemset(d_normx, 0, SIZE);
    cudaMemset(d_normy, 0, SIZE);
    cudaMemset(d_normz, 0, SIZE);
    cudaMemset(d_indicator, 0, SIZE);
    cudaMemset(d_ffx, 0, SIZE);
    cudaMemset(d_ffy, 0, SIZE);
    cudaMemset(d_ffz, 0, SIZE);

    cudaMemcpyToSymbol(CSSQ, &H_CSSQ, sizeof(float));
    cudaMemcpyToSymbol(OMEGA, &H_OMEGA, sizeof(float));
    cudaMemcpyToSymbol(SHARP_C, &H_SHARP_C, sizeof(float));
    cudaMemcpyToSymbol(SIGMA, &H_SIGMA, sizeof(float));

    cudaMemcpyToSymbol(W, &H_W, NLINKS * sizeof(float));

    cudaMemcpyToSymbol(CIX, &H_CIX, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIY, &H_CIY, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIZ, &H_CIZ, NLINKS * sizeof(int));

    //cudaMemcpyToSymbol(DATAZ, &H_DATAZ, 200 * sizeof(float));
}

