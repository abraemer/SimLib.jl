#!/bin/bash
if ! (( $# == 4 ))
then
    echo "Usage: start_run.sh GEOM N D ALPHA"
    exit
fi
sbatch --output="logs/run-ed-N_$1-$2d-alpha_$3-slurm-%j.out" slurm/run_ed.jl $1 $2 $3 $4 $(seq -1.5 0.2 1.1) 0.0 $(seq -0.5 0.025 0.3)