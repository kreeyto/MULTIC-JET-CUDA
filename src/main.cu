#include "kernels.cuh"
#include "auxFunctions.cuh"
#include "var.cuh"

int main(int argc, char* argv[]) {
    auto START_TIME = chrono::high_resolution_clock::now();
    if (argc < 3) {
        cerr << "Erro: Uso: " << argv[0] << " <velocity set> <ID>" << endl;
        return 1;
    }
    string VELOCITY_SET = argv[1];
    string ID = argv[2];

    string BASE_DIR;   
    #ifdef _WIN32
        BASE_DIR = "..\\";
    #else
        BASE_DIR = "../";
    #endif
    string MODEL_DIR = BASE_DIR + "bin/" + VELOCITY_SET + "/";
    string SIM_DIR = MODEL_DIR + ID + "/";
    
    #ifdef _WIN32
        string MKDIR_COMMAND = "mkdir \"" + SIM_DIR + "\"";
    #else
        string MKDIR_COMMAND = "mkdir -p \"" + SIM_DIR + "\"";
    #endif
    
    int ret = system(MKDIR_COMMAND.c_str());
    (void)ret;

    // ============================================================================================================================================================= //

    // ================================= //
    int MACRO_SAVE = 100, NSTEPS = 25000;
    // ================================= //
    initializeVars();

    string INFO_FILE = SIM_DIR + ID + "_info.txt";
    float H_TAU = 0.505f; // dummy value for info file
    generateSimulationInfoFile(INFO_FILE, NX, NY, NZ, MACRO_SAVE, NSTEPS, H_TAU, ID, VELOCITY_SET);

    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // STREAMS
    cudaStream_t mainStream;
    cudaStreamCreate(&mainStream);

    // ================== INIT ================== //

        initDist<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
            d_f, NX, NY, NZ
        ); 

    // ========================================= //

    vector<float> phi_host(NX * NY * NZ);
    vector<float> uz_host(NX * NY * NZ);

    for (int STEP = 0; STEP <= NSTEPS ; ++STEP) {
        cout << "Passo " << STEP << " de " << NSTEPS << " iniciado..." << endl;

        // ================= PHASE FIELD ================= //
            
            gpuPhaseField<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_phi, d_g, NX, NY, NZ
            ); 

        // =============================================== // 
        


        // ===================== INTERFACE ===================== //

            gpuGradients<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_phi, d_normx, d_normy, d_normz, 
                d_indicator, NX, NY, NZ
            ); 

            gpuCurvature<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_indicator, d_normx, d_normy, d_normz,
                d_ffx, d_ffy, d_ffz, NX, NY, NZ
            );

        // =================================================== // 

        

        // ==================== COLLISION & STREAMING ==================== //
            
            gpuMomCollisionStream<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_ux, d_uy, d_uz, d_rho, 
                d_ffx, d_ffy, d_ffz, d_f,
                NX, NY, NZ
            ); 

            gpuPhaseCollisionStream<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_g, d_ux, d_uy, d_uz, 
                d_phi, d_normx, d_normy, d_normz, 
                NX, NY, NZ
            ); 

        // =============================================================== //    


    
        // ========================================== BOUNDARY ========================================== //

            gpuInflow<<<numBlocks, threadsPerBlock, 0, mainStream>>> (
                d_rho, d_phi,
                d_ux, d_uy, d_uz, d_f, d_g, 
                d_ffx, d_ffy, d_ffz,
                U_JET, DIAM,
                NX, NY, NZ
                //STEP, MACRO_SAVE
            ); 

        // ============================================================================================= //

        cudaDeviceSynchronize();

        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(d_phi, NX * NY * NZ, SIM_DIR, ID, STEP, "phi");
            copyAndSaveToBinary(d_uz, NX * NY * NZ, SIM_DIR, ID, STEP, "uz");

            cout << "Passo " << STEP << ": Dados salvos em " << SIM_DIR << endl;
        }
    }

    cudaStreamDestroy(mainStream);

    float *pointers[] = {d_f, d_g, d_phi, d_rho,
                          d_normx, d_normy, d_normz, d_indicator,
                          d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz
                        };
    freeMemory(pointers, 14);  

    auto END_TIME = chrono::high_resolution_clock::now();
    chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    long long TOTAL_CELLS = static_cast<long long>(NX) * NY * NZ * NSTEPS;
    double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    cout << "\n// =============================================== //\n";
    cout << "     Tempo total de execução: " << ELAPSED_TIME.count() << " segundos\n";
    cout << "     Desempenho: " << MLUPS << " MLUPS\n";
    cout << "// =============================================== //\n" << endl;

    return 0;
}
