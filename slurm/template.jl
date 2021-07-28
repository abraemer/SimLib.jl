#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00 
#SBATCH --mem=1gb 
#SBATCH --cpus-per-task=48
#SBATCH --job-name=template
#SBATCH --output="logs/template-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --threads=1 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#
println("TEMPLATE.slurm")

# check ARGS length and print usage if wrong

## environment
import Pkg
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

Pkg.activate(".")
Pkg.instantiate(; io=stdout)
Pkg.status(; io=stdout)

## imports
using SimLib
using SimLib.Positions

## constants and ARGS
const PREFIX = path_prefix()
@show PREFIX

## functions

## main

println()
logmsg("Starting!")

@time begin
    ## DO STUFF

    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!