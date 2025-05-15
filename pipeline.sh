#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

GPU_ARCH=86

OS_TYPE=$(uname -s)

run_pipeline() {
    local VELOCITY_SET=$1
    local ID=$2

    if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
        echo -e "${RED}Error: Insufficient arguments. Usage: ./pipeline.sh <velocity_set> <id>${RESET}"
        exit 1
    fi

    BASE_DIR=$(cd "$(dirname "$0")" && pwd)
    MODEL_DIR="${BASE_DIR}/bin/${VELOCITY_SET}"
    SIMULATION_DIR="${MODEL_DIR}/${ID}"

    echo -e "${YELLOW}Preparing directories ${CYAN}${SIMULATION_DIR}${RESET}"
    mkdir -p "${SIMULATION_DIR}"

    echo -e "${YELLOW}Cleaning directory ${CYAN}${SIMULATION_DIR}${RESET}"
    find "${SIMULATION_DIR}" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

    FILES=$(ls -A "${SIMULATION_DIR}" | grep -v '^\.gitkeep$')
    if [ -n "$FILES" ]; then
        echo -e "${RED}Error: The directory ${CYAN}${SIMULATION_DIR}${RED} still contains files!${RESET}"
        exit 1
    else
        echo -e "${GREEN}Directory cleaned successfully.${RESET}"
    fi

    echo -e "${YELLOW}Going to ${CYAN}${BASE_DIR}${RESET}"
    cd "${BASE_DIR}" || { echo -e "${RED}Error: Directory ${CYAN}${BASE_DIR}${RED} not found!${RESET}"; exit 1; }

    echo -e "${BLUE}Executing: ${CYAN}sh compile.sh ${VELOCITY_SET} ${ID}${RESET}"
    sh compile.sh "${VELOCITY_SET}" "${ID}" || { echo -e "${RED}Error executing compile.sh${RESET}"; exit 1; }

    EXECUTABLE="${MODEL_DIR}/${ID}sim_${VELOCITY_SET}_sm${GPU_ARCH}"
    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Erro: Executable not found in ${CYAN}${EXECUTABLE}${RESET}"
        exit 1
    fi

    echo -e "${BLUE}Executando: ${CYAN}${EXECUTABLE} ${VELOCITY_SET} ${ID}${RESET}"
    if [[ "$OS_TYPE" == "Linux" ]]; then
        sudo "${EXECUTABLE}" "${VELOCITY_SET}" "${ID}" 1 || {
            echo -e "${RED}Error running the simulator${RESET}"
            exit 1
        }
    else
        "${EXECUTABLE}.exe" "${VELOCITY_SET}" "${ID}" 1 || {
            echo -e "${RED}Error running the simulator${RESET}"
            exit 1
        }
    fi

    echo -e "${YELLOW}Going to ${CYAN}${BASE_DIR}/post${RESET}"
    cd "${BASE_DIR}/post" || { echo -e "${RED}Error: Directory ${CYAN}${BASE_DIR}/post${RED} not found!${RESET}"; exit 1; }

    echo -e "${BLUE}Executing: ${CYAN}./run_post.sh ${ID} ${VELOCITY_SET}${RESET}"
    ./run_post.sh "${ID}" "${VELOCITY_SET}" || { echo -e "${RED}Error executing run_post.sh${RESET}"; exit 1; }

    echo -e "${GREEN}Process completed successfully!${RESET}"
}

run_pipeline "$1" "$2"
