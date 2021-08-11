#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00 
#SBATCH --mem=150gb 
#SBATCH --cpus-per-task=48
#SBATCH --job-name=cusp-zoom2-N_14
#SBATCH --output="logs/cusp-zoom2-N_14-%j.out"
########### End Slurm header ##########

# load modules
# not needed - julia installed locally

# ed_cusp_zoom.jl geom N dim alpha
exec julia --color=no --procs 48 --startup-file=no "slurm/cusp-zoom/ed_cusp_zoom2.jl" $1 14 1 $2