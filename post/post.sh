#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Uso: ./post.sh <ID> <velocity_set>"
    exit 1
fi

ID=$1   
VELOCITY_SET=$2

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PYTHON_CMD="python"
else
    echo "Sistema operacional não reconhecido. Tentando python por padrão."
    PYTHON_CMD="python"
fi

$PYTHON_CMD exampleVTK.py "$ID" "$VELOCITY_SET"
