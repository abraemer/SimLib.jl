#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00 
#SBATCH --mem=32gb 
#SBATCH --job-name=run-ed
#SBATCH --cpus-per-task=48
#SBATCH --output="logs/run_ed-slurm-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
exec julia --color=no --threads=8 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#

println("RUN_ED.jl")

if length(ARGS) < 5
    println("Usage: run_ed.jl geom N dim alpha field...")
    exit()
end

## environment
using Pkg
using LinearAlgebra

println("Working Directory:          $(pwd())" )
println("Running on host:            $(gethostname())" )
println("Job id:                     $(get(ENV, "SLURM_JOB_ID", ""))" )
println("Job name:                   $(get(ENV, "SLURM_JOB_NAME", ""))" )
println("Number of nodes allocated:  $(get(ENV, "SLURM_JOB_NUM_MODES", ""))" )
println("Number of cores allocated:  $(get(ENV, "SLURM_NTASKS", ""))" )
println("#threads of Julia:          $(Threads.nthreads())")
println("#threads of BLAS:           $(BLAS.get_num_threads())")

Pkg.activate(".")
Pkg.instantiate(; io=stdout)
Pkg.status(; io=stdout)

## imports
using SimLib
using SimLib.Positions
using SimLib.ED

## Constants and ARGS
const GEOMETRY = Symbol(lowercase(ARGS[1]))
const N = parse(Int, ARGS[2])
const DIM = parse(Int, ARGS[3])
const ALPHA = parse(Float64, ARGS[4])
const FIELDS = sort!(collect(Set(parse.(Float64, ARGS[5:end]))))
@show ARGS
@show GEOMETRY
@show N
@show DIM
@show ALPHA
@show FIELDS

const PREFIX = path_prefix()
@show PREFIX

## main

println()
logmsg("Starting!")

@time begin
    SimLib.logmsg("Loading position data")
    pd = Positions.load(PREFIX, GEOMETRY, N, DIM)
    SimLib.logmsg("Running ED")
    ED.save(PREFIX, ED.run_ed(pd, ALPHA, FIELDS))
    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!

# SimLib.logmsg("Loading position data")
# pd = Positions.load(PREFIX, :box, 6, 1)
# SimLib.logmsg("Running ED")
# ED.save(PREFIX, ED.run_ed(pd, 6, [0.1,0.2]))