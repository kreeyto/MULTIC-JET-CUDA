#include "auxFunctions.cuh"
#include "var.cuh"

void freeMemory(float **pointers, int COUNT) {
    for (int i = 0; i < COUNT; ++i) {
        if (pointers[i] != nullptr) {
            cudaFree(pointers[i]);
        }
    }
}

void generateSimulationInfoFile(
    const string& FILEPATH, const int NX, const int NY, const int NZ, const int STAMP, const int NSTEPS, const float TAU, 
    const string& SIM_ID, const string& VELOCITY_SET
) {
    try {
        ofstream file(FILEPATH);

        if (!file.is_open()) {
            cerr << "Erro ao abrir o arquivo: " << FILEPATH << endl;
            return;
        }

        file << "---------------------------- SIMULATION INFORMATION ----------------------------\n"
             << "                           Simulation ID: " << SIM_ID << '\n'
             << "                           Velocity set: " << VELOCITY_SET << '\n'
             << "                           Precision: " << PRECISION_TYPE << '\n'
             << "                           NX: " << NX << '\n'
             << "                           NY: " << NY << '\n'
             << "                           NZ: " << NZ << '\n'
             << "                           NZ_TOTAL: " << NZ << '\n'
             << "                           Tau: " << TAU << '\n'
             << "                           Umax: 0.000000e+00\n"
             << "                           FX: 0.000000e+00\n"
             << "                           FY: 0.000000e+00\n"
             << "                           FZ: 0.000000e+00\n"
             << "                           Save steps: " << STAMP << '\n'
             << "                           Nsteps: " << NSTEPS << '\n'
             << "                           MLUPS: 1.187970e+01\n"
             << "--------------------------------------------------------------------------------\n";

        file.close();
        cout << "Arquivo de informações da simulação criado em: " << FILEPATH << endl;
    } catch (const exception& e) {
        cerr << "Erro ao gerar o arquivo de informações: " << e.what() << endl;
    }
}

void copyAndSaveToBinary(
    const float* d_data, size_t SIZE, const string& SIM_DIR, 
    const string& ID, int STEP, const string& VAR_NAME
) {
    vector<float> host_data(SIZE);
    
    cudaMemcpy(host_data.data(), d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    ostringstream FILENAME;
    FILENAME << SIM_DIR << ID << "_" << VAR_NAME << setw(6) << setfill('0') << STEP << ".bin";
    
    ofstream file(FILENAME.str(), ios::binary);
    if (!file) {
        cerr << "Erro ao abrir o arquivo " << FILENAME.str() << " para escrita." << endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(host_data.data()), host_data.size() * sizeof(float));
    file.close();
}
