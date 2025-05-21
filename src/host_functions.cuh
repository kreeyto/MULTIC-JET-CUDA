#pragma once
#include "common.cuh"

__host__ __forceinline__ std::string createSimulationDirectory(
    const std::string& VELOCITY_SET, const std::string& SIM_ID
) {
    std::string BASE_DIR = 
    #ifdef _WIN32
        ".\\";
    #else
        "./";
    #endif

    std::string SIM_DIR = BASE_DIR + "bin/" + VELOCITY_SET + "/" + SIM_ID + "/";
    
    #ifdef _WIN32
        std::string MKDIR_COMMAND = "mkdir \"" + SIM_DIR + "\"";
    #else
        std::string MKDIR_COMMAND = "mkdir -p \"" + SIM_DIR + "\"";
    #endif

    int ret = std::system(MKDIR_COMMAND.c_str());
    (void)ret;

    return SIM_DIR;
}

__host__ __forceinline__ void computeAndPrintOccupancy() {
    int minGridSize = 0, blockSize = 0;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, gpuMomCollisionStream, 0, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error in calculating occupancy: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, gpuMomCollisionStream, blockSize, 0);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Optimal block size       : " << blockSize << "\n";
    std::cout << "     Minimum grid size        : " << minGridSize << "\n";
    std::cout << "     Active blocks per SM     : " << maxBlocksPerSM << "\n";
    std::cout << "// =============================================== //\n" << std::endl;
}

__host__ __forceinline__ void generateSimulationInfoFile(
    const std::string& SIM_DIR, const std::string& SIM_ID, const std::string& VELOCITY_SET, 
    const int NSTEPS, const int MACRO_SAVE, 
    const float TAU, const double MLUPS
) {
    std::string INFO_FILE = SIM_DIR + SIM_ID + "_info.txt";
    try {
        std::ofstream file(INFO_FILE);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << INFO_FILE << std::endl;
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
        std::cout << "Simulation information file created in: " << INFO_FILE << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error generating information file: " << e.what() << std::endl;
    }
}

__host__ __forceinline__ void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const std::string& SIM_DIR, 
    const std::string& ID, int STEP, const std::string& VAR_NAME
) {
    std::vector<float> host_data(SIZE);

    checkCudaErrors(cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    std::ostringstream FILENAME;
    FILENAME << SIM_DIR << ID << "_" << VAR_NAME << std::setw(6) << std::setfill('0') << STEP << ".bin";

    std::ofstream file(FILENAME.str(), std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << FILENAME.str() << " for writing." << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(float));
    file.close();
}