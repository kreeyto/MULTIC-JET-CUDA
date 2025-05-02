#!/bin/bash

CC=86
VELOCITY_SET=$1
ID=$2

if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
    echo "Uso: ./compile.sh <VELOCITY_SET> <ID>"
    exit 1
fi

BASE_DIR=$(dirname "$0")
SRC_DIR="${BASE_DIR}/src"
OUTPUT_DIR="${BASE_DIR}/bin/${VELOCITY_SET}"
EXECUTABLE="${OUTPUT_DIR}/${ID}sim_${VELOCITY_SET}_sm${CC}"

mkdir -p "${OUTPUT_DIR}"

echo "Compilando para ${EXECUTABLE}..."

nvcc -O3 --restrict \
     -gencode arch=compute_${CC},code=sm_${CC} \
     -rdc=true --ptxas-options=-v \
     -I"${SRC_DIR}" \
     "${SRC_DIR}/core/main.cu" \
     "${SRC_DIR}/core/lbm.cu" \
     "${SRC_DIR}/device/setup.cu" \
     "${SRC_DIR}/host/io.cu" \
     -lcudadevrt -lcurand -D${VELOCITY_SET} \
     -o "${EXECUTABLE}"

if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Erro na compilação!"
    exit 1
fi
