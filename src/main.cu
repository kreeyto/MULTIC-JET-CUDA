#include "kernels.cuh"
#include "hostFunctions.cuh"
#include "deviceStructs.cuh"

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
        BASE_DIR = ".\\";
    #else
        BASE_DIR = "./";
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

    initDeviceVars();

    string INFO_FILE = SIM_DIR + ID + "_info.txt";
    generateSimulationInfoFile(INFO_FILE, MACRO_SAVE, NSTEPS, H_TAU, ID, VELOCITY_SET);

    dim3 threadsPerBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    dim3 threadsPerBlockBC(BLOCK_SIZE_X*2,BLOCK_SIZE_Y*2);  
    dim3 numBlocksBC((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);    

    cudaStream_t mainStream;
    checkCudaErrors(cudaStreamCreate(&mainStream));

    gpuInitDistributions<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
    getLastCudaError("gpuInitDistributions");

    vector<float> phi_host(NX * NY * NZ); 
    vector<float> uz_host(NX * NY * NZ);

    for (int STEP = 0; STEP <= NSTEPS ; ++STEP) {
        cout << "Passo " << STEP << " de " << NSTEPS << " iniciado..." << endl;

        // ======================= INTERFACE ======================= //

            gpuComputePhaseField<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuComputePhaseField");

            gpuComputeInterface<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuComputeInterface");

        // ======================================================== // 

        

        // ==================== COLLISION & STREAMING ==================== //
            
            gpuFusedCollisionStream<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuFusedCollisionStream");

            gpuEvolveScalarField<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuEvolveScalarField");

        // =============================================================== //    


    
        // ========================================== BOUNDARY ========================================== //

            gpuApplyInflowBoundary<<<numBlocksBC,threadsPerBlockBC,0,mainStream>>> (lbm,STEP); 
            getLastCudaError("gpuApplyInflowBoundary");

        // ============================================================================================= //

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(lbm.phi, NX * NY * NZ, SIM_DIR, ID, STEP, "phi");
            copyAndSaveToBinary(lbm.uz, NX * NY * NZ, SIM_DIR, ID, STEP, "uz");

            cout << "Passo " << STEP << ": Dados salvos em " << SIM_DIR << endl;
        }
    }

    checkCudaErrors(cudaStreamDestroy(mainStream));

    cudaFree(lbm.f);
    cudaFree(lbm.g);
    cudaFree(lbm.phi);
    cudaFree(lbm.rho);
    cudaFree(lbm.ux);
    cudaFree(lbm.uy);
    cudaFree(lbm.uz);
    cudaFree(lbm.normx);
    cudaFree(lbm.normy);
    cudaFree(lbm.normz);
    cudaFree(lbm.ffx);
    cudaFree(lbm.ffy);
    cudaFree(lbm.ffz);

    auto END_TIME = chrono::high_resolution_clock::now();
    chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    long long TOTAL_CELLS = static_cast<long long>(NX) * NY * NZ * NSTEPS;
    double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    cout << "\n// =============================================== //\n";
    cout << "     Tempo total de execução: " << ELAPSED_TIME.count() << " segundos\n";
    cout << "     Desempenho: " << MLUPS << " MLUPS\n";
    cout << "// =============================================== //\n" << endl;

    getLastCudaError("Final sync");
    return 0;
}
