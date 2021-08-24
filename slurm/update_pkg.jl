#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=1gb
#SBATCH --job-name=update-pkg
#SBATCH --output="logs/update-pkg-slurm-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --threads=1 --startup-file=no "${BASH_SOURCE[0]}" "$@"
=#
println("PKG-UPDATE.slurm")

# check ARGS length and print usage if wrong

## environment
using LinearAlgebra # for BLAS threads

println("Working Directory:          $(pwd())" )
println("Running on host:            $(gethostname())" )
println("Job id:                     $(get(ENV, "SLURM_JOB_ID", ""))" )
println("Job name:                   $(get(ENV, "SLURM_JOB_NAME", ""))" )
println("Number of nodes allocated:  $(get(ENV, "SLURM_JOB_NUM_MODES", ""))" )
println("Number of cores allocated:  $(get(ENV, "SLURM_NTASKS", ""))" )
println("#threads of Julia:          $(Threads.nthreads())")
println("#threads of BLAS:           $(BLAS.get_num_threads())")
@show ARGS

import Pkg
Pkg.activate(".")
println("\nUpdate")
@time Pkg.update(; io=stdout);
@time Pkg.update("XXZNumerics"; io=stdout);

println("\nPrecompile")
Pkg.precompile(; io=stdout)
Pkg.status(; io=stdout)
