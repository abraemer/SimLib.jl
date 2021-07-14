#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00 
#SBATCH --mem=20gb 
#SBATCH --cpus-per-task=48
#SBATCH --job-name=blas-threads
#SBATCH --output="logs/blas-threads-slurm-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

exec julia --color=no --threads=1 --startup-file=no "${BASH_SOURCE[0]}" "$@" 
=#
println("blas_threads.jl")

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
using XXZNumerics
using SimLib
using SimLib.Positions
using Printf
using JLD2: jldsave

## constants and ARGS
const PREFIX = path_prefix()
@show PREFIX

## functions

make_matrix(size) = Hermitian(Matrix(xxzmodel(size, 1, -0.73) + 0.25*fieldterm(size, σx)))

## main

println()
# dry run
println()
logmsg("Starting!")

Nrange = 6:15
tcrange = [1,2,3,4,6,8,12,16]

times = zeros(Float64, length(Nrange), length(tcrange))

eigen(make_matrix(3))# warmup

@time begin
    for (i, N) in enumerate(Nrange)
        m = make_matrix(N)
        for (j, tc) in enumerate(tcrange)
            BLAS.set_num_threads(tc)
            logmsg(@sprintf("N=%02i - threadcount=%02i", N, BLAS.get_num_threads()))
            start = time()
            @time eigen(m);
            times[i,j] = time()-start
            GC.gc()
        end
    end
    jldsave(joinpath(path_prefix(), "blas_thread_timings.jld2"); data=times, N=Nrange, threads=tcrange)
    logmsg("Done!")
end
## REMEMBER TO SET RESOURCE HEADER FOR SLURM!