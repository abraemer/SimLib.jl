#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00 
#SBATCH --mem=1gb 
#SBATCH --job-name=positions
#SBATCH --output="logs/pos-slurm-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --threads=1 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#
using LinearAlgebra # for BLAS threads

println("CREATE_POSITIONS.slurm")
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
Pkg.instantiate(; io=stdout)
Pkg.status(; io=stdout)

## imports
using SimLib
using SimLib.Positions

## Constants and ARGS

const PREFIX = try
        joinpath(readchomp(`ws_find cusp`), "julia")
    catch e
        joinpath(pwd(), "data")
    end
@show PREFIX

## Functions

## main

println()
logmsg("Starting!")

@time begin
    ## DO STUFF

    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!