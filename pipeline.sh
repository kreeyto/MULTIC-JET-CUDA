#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

OS_TYPE=$(uname -s)

if [[ "$OS_TYPE" == "Linux" ]]; then
    BASE_DIR=$(dirname "$(readlink -f "$0")")

    VELOCITY_SET=$1
    ID=$2

    if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
        echo -e "${RED}Erro: Argumentos insuficientes. Uso: ./pipeline.sh <velocity_set> <id>${RESET}"
        exit 1
    fi

    MODEL_DIR=$BASE_DIR/bin/${VELOCITY_SET}
    SIMULATION_DIR=${MODEL_DIR}/${ID}
    echo -e "${YELLOW}Preparando os diretórios ${CYAN}${SIMULATION_DIR}${RESET}"
    mkdir -p ${SIMULATION_DIR}

    echo -e "${YELLOW}Limpando o diretório ${CYAN}${SIMULATION_DIR}${RESET}"
    find "${SIMULATION_DIR}" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

    FILES=$(ls -A ${SIMULATION_DIR} | grep -v '^\.gitkeep$')
    if [ -n "$FILES" ]; then
        echo -e "${RED}Erro: O diretório ${CYAN}${SIMULATION_DIR}${RED} ainda contém arquivos!${RESET}"
        exit 1
    else
        echo -e "${GREEN}Diretório limpo com sucesso.${RESET}"
    fi

    echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
    cd $BASE_DIR/src || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

    echo -e "${BLUE}Executando: ${CYAN}sh compile.sh ${VELOCITY_SET} ${ID}${RESET}"
    sh compile.sh ${VELOCITY_SET} ${ID} || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

    EXECUTABLE=$(realpath "${MODEL_DIR}/${ID}sim_${VELOCITY_SET}_sm86")

    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Erro: Executável não encontrado em ${CYAN}${EXECUTABLE}${RESET}"
        exit 1
    fi

    echo -e "${BLUE}Executando: ${CYAN}sudo ${EXECUTABLE} ${VELOCITY_SET} ${ID}${RESET}"
    sudo "${EXECUTABLE}" "${VELOCITY_SET}" "${ID}" 1 || {
        echo -e "${RED}Erro na execução do simulador${RESET}"
        exit 1
    }

    echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/post${RESET}"
    cd $BASE_DIR/post || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/post${RED} não encontrado!${RESET}"; exit 1; }

    echo -e "${BLUE}Executando: ${CYAN}./post.sh ${ID} ${VELOCITY_SET}${RESET}"
    ./post.sh ${ID} ${VELOCITY_SET} || { echo -e "${RED}Erro na execução do script post.sh${RESET}"; exit 1; }

    echo -e "${GREEN}Processo concluído com sucesso!${RESET}"
    
elif [[ "$OS_TYPE" == "MINGW64_NT"* || "$OS_TYPE" == "MSYS_NT"* || "$OS_TYPE" == "CYGWIN_NT"* ]]; then

    BASE_DIR=$(cd "$(dirname "$0")" && pwd)

    VELOCITY_SET=$1
    ID=$2

    if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
        echo -e "${RED}Erro: Argumentos insuficientes. Uso: ./pipeline.sh <velocity_set> <id>${RESET}"
        exit 1
    fi

    MODEL_DIR=$BASE_DIR/bin/${VELOCITY_SET}
    SIMULATION_DIR=${MODEL_DIR}/${ID}

    echo -e "${YELLOW}Preparando os diretórios ${CYAN}${SIMULATION_DIR}${RESET}"
    mkdir -p "${SIMULATION_DIR}"

    echo -e "${YELLOW}Limpando o diretório ${CYAN}${SIMULATION_DIR}${RESET}"
    rm -rf "${SIMULATION_DIR:?}/"*

    if [ -n "$(ls -A "${SIMULATION_DIR}" 2>/dev/null)" ]; then
        echo -e "${RED}Erro: O diretório ${CYAN}${SIMULATION_DIR}${RED} ainda contém arquivos!${RESET}"
        exit 1
    else
        echo -e "${GREEN}Diretório limpo com sucesso.${RESET}"
    fi

    echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
    cd "$BASE_DIR/src" || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

    echo -e "${BLUE}Executando: ${CYAN}sh compile.sh ${VELOCITY_SET} ${ID}${RESET}"
    sh compile.sh "${VELOCITY_SET}" "${ID}" || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

    EXECUTABLE="${MODEL_DIR}/${ID}sim_${VELOCITY_SET}_sm86"

    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Erro: Executável não encontrado em ${CYAN}${EXECUTABLE}${RESET}"
        exit 1
    fi

    echo -e "${BLUE}Executando: ${CYAN}${EXECUTABLE} ${VELOCITY_SET} ${ID}${RESET}"
    "${EXECUTABLE}".exe "${VELOCITY_SET}" "${ID}" 1 || {
        echo -e "${RED}Erro na execução do simulador${RESET}"
        exit 1
    }

    echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/post${RESET}"
    cd "$BASE_DIR/post" || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/post${RED} não encontrado!${RESET}"; exit 1; }

    echo -e "${BLUE}Executando: ${CYAN}./post.sh ${ID} ${VELOCITY_SET}${RESET}"
    ./post.sh "${ID}" "${VELOCITY_SET}" || { echo -e "${RED}Erro na execução do script post.sh${RESET}"; exit 1; }

    echo -e "${GREEN}Processo concluído com sucesso!${RESET}"

else
    echo -e "${RED}Sistema operacional não suportado.${RESET}"
    exit 1
fi
