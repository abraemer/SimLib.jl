#!/bin/bash
if ! (( $# == 4 ))
then
    echo "Usage: start_run.sh GEOM N D ALPHA"
    exit
fi
sbatch --output="logs/run-ed-$1-N_$2-$3d-alpha_$4-slurm-%j.out" slurm/run_ed.jl $1 $2 $3 $4 $(seq -1.5 0.2 1.1) 0.0 $(seq -0.5 0.025 0.3)