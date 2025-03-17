#!/bin/bash

CC=86

VELOCITY_SET=$1
ID=$2

BASE_DIR=$(dirname "$0")/..
SRC_DIR=${BASE_DIR}/src
OUTPUT_DIR=${BASE_DIR}/bin/${VELOCITY_SET}
EXECUTABLE_NAME="${ID}sim_${VELOCITY_SET}_sm${CC}"

mkdir -p ${OUTPUT_DIR}

echo "Compilando para ${OUTPUT_DIR}/${EXECUTABLE_NAME}..."

nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v --restrict \
    ${SRC_DIR}/*.cu \
    -lcudadevrt -lcurand -D${VELOCITY_SET} \
    -o ${OUTPUT_DIR}/${EXECUTABLE_NAME}

if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Erro na compilação!"
    exit 1
fi
