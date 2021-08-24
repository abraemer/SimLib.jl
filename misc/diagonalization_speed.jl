#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=10gb
#SBATCH --cpus-per-task=48
#SBATCH --job-name=diag-test
#SBATCH --output="logs/diag-test-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
=#
println("diagonalization_speed.jl")

# check ARGS length and print usage if wrong


## environment
import Pkg
using LinearAlgebra # for BLAS threads

BLAS.set_num_threads(96)

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
using LinearAlgebra

using XXZNumerics

## constants and ARGS
const PREFIX = joinpath(path_prefix(), "cusp-zoom2-test")
@show PREFIX
const GEOMETRY = :box_pbc
const DIM = 1
const N = 15
const ALPHA = 6
const FIELDS = collect(-0.11:0.005:-0.05)
const ρs = [1.0]
const SHOTS = 2#200

@show GEOMETRY
@show DIM
@show N
@show ALPHA
@show FIELDS
@show ρs
@show SHOTS

const LOCATION = SaveLocation(; prefix=PREFIX)
## functions

## main

println()
logmsg("Starting!")

@time begin
    ## DO STUFF
    ensdd = EnsembleDataDescriptor(GEOMETRY, DIM, N, ALPHA, SHOTS, ρs, FIELDS, :ensembles, zbasis(N), LOCATION)
    logmsg("Computing $ensdd")
    save(create(ensdd); suffix="p_1")
    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!
