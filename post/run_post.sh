#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./post.sh <ID> <velocity_set>"
    exit 1
fi

ID=$1   
VELOCITY_SET=$2

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PYTHON_CMD="python"
else
    echo "Operating system not recognized. Trying python by default."
    PYTHON_CMD="python"
fi

$PYTHON_CMD process_steps.py "$ID" "$VELOCITY_SET"
