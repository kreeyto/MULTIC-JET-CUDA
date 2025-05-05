#include "core/kernels.cuh"
#include "host/io.cuh"
#include "device/data.cuh"

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

    initializeVars();

    string INFO_FILE = SIM_DIR + ID + "_info.txt";
    generateSimulationInfoFile(INFO_FILE, MACRO_SAVE, NSTEPS, H_TAU, ID, VELOCITY_SET);

    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    dim3 threadsPerBlockBC(16,16);  
    dim3 numBlocksBC((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);    

    cudaStream_t mainStream;
    checkCudaErrors(cudaStreamCreate(&mainStream));

    gpuInitDistributions<<<numBlocks,threadsPerBlock,0,mainStream>>> (d); 
    getLastCudaError("gpuInitDistributions");

    //int haloX = threadsPerBlock.x + 2;
    //int haloY = threadsPerBlock.y + 2;
    //int haloZ = threadsPerBlock.z + 2;
    //size_t sharedSize = 3 * haloX * haloY * haloZ * sizeof(float);

    vector<float> phi_host(NX * NY * NZ); 
    vector<float> uz_host(NX * NY * NZ);

    for (int STEP = 0; STEP <= NSTEPS ; ++STEP) {
        cout << "Passo " << STEP << " de " << NSTEPS << " iniciado..." << endl;

        // ================= PHASE & INTERFACE ================= //

            gpuComputePhaseField<<<numBlocks,threadsPerBlock,0,mainStream>>> (d); 
            getLastCudaError("gpuComputePhaseField");

            gpuComputeInterface<<<numBlocks,threadsPerBlock,0,mainStream>>> (d); 
            getLastCudaError("gpuComputeInterface");

        // ===================================================== // 

        

        // ==================== COLLISION & STREAMING ==================== //
            
            gpuFusedCollisionStream<<<numBlocks,threadsPerBlock,0,mainStream>>> (d); 
            getLastCudaError("gpuFusedCollisionStream");

            gpuEvolveScalarField<<<numBlocks,threadsPerBlock,0,mainStream>>> (d); 
            getLastCudaError("gpuEvolveScalarField");

        // =============================================================== //    


    
        // ========================================== BOUNDARY ========================================== //

            gpuApplyInflowBoundary<<<numBlocksBC,threadsPerBlockBC,0,mainStream>>> (d,STEP); 
            getLastCudaError("gpuApplyInflowBoundary");

        // ============================================================================================= //

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(d.phi, NX * NY * NZ, SIM_DIR, ID, STEP, "phi");
            copyAndSaveToBinary(d.uz, NX * NY * NZ, SIM_DIR, ID, STEP, "uz");

            cout << "Passo " << STEP << ": Dados salvos em " << SIM_DIR << endl;
        }
    }

    checkCudaErrors(cudaStreamDestroy(mainStream));

    cudaFree(d.f);
    cudaFree(d.g);
    cudaFree(d.rho);
    cudaFree(d.phi);
    cudaFree(d.ux);
    cudaFree(d.uy);
    cudaFree(d.uz);
    cudaFree(d.normx);
    cudaFree(d.normy);
    cudaFree(d.normz);
    cudaFree(d.ffx);
    cudaFree(d.ffy);
    cudaFree(d.ffz);

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
