#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00 
#SBATCH --mem=90gb 
#SBATCH --cpus-per-task=48
#SBATCH --job-name=field-section
#SBATCH --output="logs/field-section-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --procs 50 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#
println("ed_field_section.jl")

# check ARGS length and print usage if wrong
if length(ARGS) != 5
    println("Usage: ed_field_section.jl geom N dim alpha field")
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

using XXZNumerics
import SimLib.Positions
import SimLib.ED

## constants and ARGS
const PREFIX = joinpath(path_prefix(), "field-section")
@show PREFIX
const GEOMETRY = Symbol(lowercase(ARGS[1]))
const N = parse(Int, ARGS[2])
const DIM = parse(Int, ARGS[3])
const ALPHA = parse(Float64, ARGS[4])
const FIELD = parse(Float64, ARGS[5])
const ρs = [0.5:0.05:1.95..., 1.99]
const SHOTS = 100

@show GEOMETRY
@show N
@show DIM
@show ALPHA
@show FIELD
@show ρs
@show SHOTS

## functions
function createAndSave(shots, geom, N, dim, ρs)
    logmsg("Creating Position data with $shots shots, $geom N=$N $(dim)d and $(length(ρs))")
    data = Positions.PositionData(geom, ρs, shots, N, dim)
    Positions.save(PREFIX, Positions.create_positions!(data))
    data
end

function preparePosdata(geom, N, dim, ρs)
    p = Positions.position_datapath(PREFIX, geom, N, dim)
    if !isfile(p)
        logmsg("No position data found!")
        createAndSave(SHOTS, geom, N, dim, ρs)
    else
        logmsg("Found existing position data!")
        data = Positions.load(p)
        if Positions.ρs(data) == ρs
            data
        else
            logmsg("Does not match the required densities: generating anew.")
            createAndSave(SHOTS, geom, N, dim, ρs)
        end
    end
end


## main

println()
logmsg("Starting!")

@time begin
    ## DO STUFF
    logmsg("Preparing position data")
    posdata = preparePosdata(GEOMETRY, N, DIM, ρs)
    logmsg("Running ED")
    eddata = ED.run_ed_parallel2(posdata, ALPHA, [FIELD])
    logmsg("Saving")
    ED.save(PREFIX, eddata; suffix="-field_$FIELD")
    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!