#!/bin/bash

CC=86

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3

BASE_DIR=$(dirname "$0")/..
SRC_DIR=${BASE_DIR}/src
OUTPUT_DIR=${BASE_DIR}/bin/${FLUID_MODEL}_${PHASE_MODEL}
EXECUTABLE_NAME="${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm${CC}"

mkdir -p ${OUTPUT_DIR}

echo "Compilando para ${OUTPUT_DIR}/${EXECUTABLE_NAME}..."

nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v --restrict \
    ${SRC_DIR}/*.cu \
    -lcudadevrt -lcurand -D${FLUID_MODEL} -D${PHASE_MODEL} \
    -o ${OUTPUT_DIR}/${EXECUTABLE_NAME}

if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Erro na compilação!"
    exit 1
fi
