#!/bin/bash

CC=86
VELOCITY_SET=$1
ID=$2

if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
    echo "Usage: ./compile.sh <VELOCITY_SET> <ID>"
    exit 1
fi

BASE_DIR=$(dirname "$0")
SRC_DIR="${BASE_DIR}/src"
OUTPUT_DIR="${BASE_DIR}/bin/${VELOCITY_SET}"
EXECUTABLE="${OUTPUT_DIR}/${ID}sim_${VELOCITY_SET}_sm${CC}"

mkdir -p "${OUTPUT_DIR}"

echo "Compiling to ${EXECUTABLE}..."

nvcc -O3 --restrict \
     -gencode arch=compute_${CC},code=sm_${CC} \
     -rdc=true --ptxas-options=-v \
     -I"${SRC_DIR}" \
     "${SRC_DIR}/main.cu" \
     "${SRC_DIR}/lbm_int.cu" \
     "${SRC_DIR}/lbm.cu" \
     "${SRC_DIR}/lbm_bcs.cu" \
     "${SRC_DIR}/device_setup.cu" \
     -lcudadevrt -lcurand -maxrregcount=91 -D${VELOCITY_SET} \
     -o "${EXECUTABLE}"

if [ $? -eq 0 ]; then
    echo "Compilation completed successfully: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Compilation error!"
    exit 1
fi
