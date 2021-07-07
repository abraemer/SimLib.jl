#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00 
#SBATCH --mem=1gb 
#SBATCH --job-name=positions
#SBATCH --output="logs/pos-slurm-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --threads=1 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#
using Pkg
using LinearAlgebra

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

Pkg.activate(".")
Pkg.status()

using Random
using SimLib
using SimLib.Positions

const SHOTS = length(ARGS) > 0 ? parse(Int64, ARGS[1]) : 100
const RHO_SAMPLES = 20
const PREFIX = try
        joinpath(run(`ws_find cusp`), "julia")
    catch e
        joinpath(pwd(), "data")
    end
@show PREFIX

function create(geom, N, dim)
    rho_ranges = SimLib.SAFE_RHO_RANGES[geom]
    if dim > length(rho_ranges)
        return
    end
    (rho_start, rho_end) = rho_ranges[dim]
    rho_step = (rho_end - rho_start)/(RHO_SAMPLES-1)
    logmsg("geometry=$geom N=$N dim=$dim")
    data = PositionData(geom, rho_start:rho_step:rho_end, SHOTS, N, dim)
    save(PREFIX, create_positions!(data))
end

println()
logmsg("Starting!")
Random.seed!(5)

Threads.@threads for (geom, N, dim) in vec(collect(Iterators.product(SimLib.GEOMETRIES, 6:20, 1:3)))
    create(geom, N, dim)
end

logmsg("Done!")