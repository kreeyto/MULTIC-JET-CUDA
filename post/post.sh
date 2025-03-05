#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Uso: ./post.sh <ID> <fluid_model> <phase_model>"
    exit 1
fi

ID=$1   
FLUID_MODEL=$2
PHASE_MODEL=$3

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PYTHON_CMD="python"
else
    echo "Sistema operacional não reconhecido. Tentando python por padrão."
    PYTHON_CMD="python"
fi

$PYTHON_CMD exampleVTK.py "$ID" "$FLUID_MODEL" "$PHASE_MODEL"
