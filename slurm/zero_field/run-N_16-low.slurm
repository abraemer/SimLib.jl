#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00 
#SBATCH --mem=350gb 
#SBATCH --cpus-per-task=48
#SBATCH --job-name=zero-field
#SBATCH --output="logs/zero-field-low-%j.out"
########### End Slurm header ##########

if ! (( $# == 1 ))
then
    echo "Usage: run.sh ALPHA"
    exit
fi

exec julia --color=no --procs 48 --startup-file=no "slurm/zero_field/ed_zero_field_low_density.jl" box_pbc 1 16 $1
#exec julia --color=no --procs 4 --startup-file=no "slurm/zero_field/ed_zero_field_low_density.jl" box_pbc 1 7 $1