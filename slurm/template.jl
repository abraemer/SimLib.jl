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
println("TEMPLATE.slurm")# TODO

# check ARGS length and print usage if wrong
if length(ARGS) < 1
    println("Usage: TEMPLATE.jl ARGS") # TODO
    exit()
end

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

## Distributed setup
using Distributed
if nprocs() == 1 # we run local so go easy on amount of workers
    addprocs(4; topology=:master_worker)
end

println("#workers:                   $(nprocs())")

# using Distributed # not needed when started with -p
@everywhere import Pkg
@everywhere Pkg.activate(".")
Pkg.instantiate(; io=stdout)
Pkg.status(; io=stdout)

## imports
@everywhere using SimLib
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(2)

using XXZNumerics

## constants and ARGS
const PREFIX = joinpath(path_prefix(), "subfolder")# TODO
@show PREFIX
# const GEOMETRY = Symbol(lowercase(ARGS[1]))
# const DIM = parse(Int, ARGS[2])
# const N = parse(Int, ARGS[3])
# const ALPHA = parse(Float64, ARGS[4])
# const ρs = [...]
# const SHOTS = 100
# const BLOCK = div(N-1,2)

@show GEOMETRY
@show N
@show DIM
@show ALPHA
@show ρs
@show SHOTS
@show BLOCK

const LOCATION = SaveLocation(;prefix=PREFIX)

## main

println()
logmsg("Starting!")

@time begin
    ## DO STUFF
end
logmsg("Done!")
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!
