#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00 
#SBATCH --mem=25gb
#SBATCH --cpus-per-task=48
#SBATCH --job-name=cusp-zoom-N_12
#SBATCH --output="logs/cusp-zoom-N_12-%j.out"
########### End Slurm header ##########

# load modules
# not needed - julia installed locally

# ed_cusp_zoom.jl geom N dim alpha
exec julia --color=no --procs 48 --startup-file=no "slurm/cusp-zoom/ed_cusp_zoom.jl" "$1 12 1 $2" 