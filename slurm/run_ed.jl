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
export MKL_NUM_THREADS=96
export OMP_NUM_THREADS=96
exec julia --color=no --procs 50 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
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

using Distributed
if nprocs() == 1 # we run local so go easy on amount of workers
    addprocs(4; topology=:master_worker)
end

# using Distributed # not needed when started with -p
@everywhere import Pkg
@everywhere Pkg.activate(".")
Pkg.instantiate(; io=stdout)
Pkg.status(; io=stdout)

## imports
@everywhere using SimLib
@everywhere using LinearAlgebra
@everywhere LinearAlgebra.BLAS.set_num_threads(2)

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
    eddata = ED.run_ed_parallel2(pd, ALPHA, FIELDS)
    SimLib.logmsg("Saving")
    ED.save(PREFIX, eddata)
    SimLib.logmsg("Ensemble prediction")
    SimLib.Ensembles.save(PREFIX, SimLib.Ensembles.ensemple_predictions(eddata))
    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!

# SimLib.logmsg("Loading position data")
# pd = Positions.load(PREFIX, :box, 6, 1)
# SimLib.logmsg("Running ED")
# ED.save(PREFIX, ED.run_ed(pd, 6, [0.1,0.2]))

## NOTES!
# BLAS threads are per process. So in order to parallelize diagonalization I need to have multiple julia processes
# For this one uses Distributed. The structure should look roughly like
# 1) load Distributed
# 2) use Distributed.addprocs(N) to initialize workers (or ensure at least that there are enough - 
#    Julia can be started with -p to add processes from the start)
# 3) load libraries on ALL processes with @everywhere
# 4) Allocate SharedArrays.SharedArray for the output data
# use @spawnat :any?? or @async remote_do ?? What is the difference?
# 5) put whole computation in @sync and start subtasks with @spawnat :any (should be round-robin I think?)
#    A subtask basically performs a diagonalization, computes the results and puts them into the SharedArrays
#    Thus I do not need to do any handling of output of these subtasks. Just start and at the end of @sync
#    everything will be inside the SharedArray
# 6) Convert SharedArray to normal Array and carry on.
#
# Note: Input data does not need to be a SharedArray I think
