#pragma once
#include "common.cuh"

__host__ __forceinline__ string createSimulationDirectory(
    const string& VELOCITY_SET, const string& SIM_ID
) {
    string BASE_DIR = 
    #ifdef _WIN32
        ".\\";
    #else
        "./";
    #endif

    string SIM_DIR = BASE_DIR + "bin/" + VELOCITY_SET + "/" + SIM_ID + "/";
    
    #ifdef _WIN32
        string MKDIR_COMMAND = "mkdir \"" + SIM_DIR + "\"";
    #else
        string MKDIR_COMMAND = "mkdir -p \"" + SIM_DIR + "\"";
    #endif

    int ret = system(MKDIR_COMMAND.c_str());
    (void)ret;

    return SIM_DIR;
}

__host__ __forceinline__ void computeAndPrintOccupancy() {
    int minGridSize = 0, blockSize = 0;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, gpuFusedCollisionStream, 0, 0);
    if (err != cudaSuccess) {
        cerr << "Error in calculating occupancy: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, gpuFusedCollisionStream, blockSize, 0);

    cout << "\n// =============================================== //\n";
    cout << "     Optimal block size       : " << blockSize << "\n";
    cout << "     Minimum grid size        : " << minGridSize << "\n";
    cout << "     Active blocks per SM     : " << maxBlocksPerSM << "\n";
    cout << "// =============================================== //\n" << endl;
}

__host__ __forceinline__ void generateSimulationInfoFile(
    const string& SIM_DIR, const string& SIM_ID, const string& VELOCITY_SET, 
    const int NSTEPS, const int MACRO_SAVE, 
    const float TAU, const double MLUPS
) {
    string INFO_FILE = SIM_DIR + SIM_ID + "_info.txt";
    try {
        ofstream file(INFO_FILE);

        if (!file.is_open()) {
            cerr << "Error opening file: " << INFO_FILE << endl;
            return;
        }

        file << "---------------------------- SIMULATION INFORMATION ----------------------------\n"
             << "                           Simulation ID: " << SIM_ID << '\n'
             << "                           Velocity set: " << VELOCITY_SET << '\n'
             << "                           Precision: float\n"
             << "                           NX: " << NX << '\n'
             << "                           NY: " << NY << '\n'
             << "                           NZ: " << NZ << '\n'
             << "                           NZ_TOTAL: " << NZ << '\n'
             << "                           Tau: " << TAU << '\n'
             << "                           Umax: " << U_JET << '\n'
             << "                           Save steps: " << MACRO_SAVE << '\n'
             << "                           Nsteps: " << NSTEPS << '\n'
             << "                           MLUPS: " << MLUPS << '\n'
             << "--------------------------------------------------------------------------------\n";

        file.close();
        cout << "Simulation information file created in: " << INFO_FILE << endl;
    } catch (const exception& e) {
        cerr << "Error generating information file: " << e.what() << endl;
    }
}

__host__ __forceinline__ void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const string& SIM_DIR, 
    const string& ID, int STEP, const string& VAR_NAME
) {
    vector<float> host_data(SIZE);

    checkCudaErrors(cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    ostringstream FILENAME;
    FILENAME << SIM_DIR << ID << "_" << VAR_NAME << setw(6) << setfill('0') << STEP << ".bin";

    ofstream file(FILENAME.str(), ios::binary);
    if (!file) {
        cerr << "Error opening file " << FILENAME.str() << " for writing." << endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(float));
    file.close();
}