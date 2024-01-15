#!/bin/bash

program_name=$1

smi_output=$(nvidia-smi --query-gpu=compute_cap --format=csv)

# Parse compute capability
f_compute_capability=$(echo $smi_output | cut -d' ' -f2)
compute_capability=$(echo "$f_compute_capability" | sed 's/\.//g')


# Get rest of arguments
shift
rest_of_args=("$@")

# If rest_of_args does not contain -g, -G or -DDEBUG
if [[ ! " ${rest_of_args[@]} " =~ " -g " ]] && [[ ! " ${rest_of_args[@]} " =~ " -G " ]] && [[ ! " ${rest_of_args[@]} " =~ " -DDEBUG " ]]; then
    # Add optimization flags
    rest_of_args+=("-O3 --use_fast_math")
fi

# If path does not contain /usr/local/cuda/bin
if [[ ! "$PATH" =~ "/usr/local/cuda/bin" ]]; then
    # Add /usr/local/cuda/bin to path
    export PATH=/usr/local/cuda/bin:$PATH
fi

# Compile the program
nvcc "$program_name".cu -o $program_name -arch=sm_$compute_capability -maxrregcount=64 --extended-lambda ${rest_of_args[@]}
