using SimLib
using XXZNumerics
using Test
using Distributed
using Random

to_file(func, path; mode="w") = open(path, mode) do out
    redirect_stdout(out) do
        func()
    end
end

log_contains(path, str) = any(l -> contains(l, str), eachline(path))

const PREFIX = tempname()
mkdir(PREFIX)
@show PREFIX

Random.seed!(5)

@show pwd()

@testset "SimLib.jl" begin
    # Write your tests here.
    include("positions.jl")
    include("ed.jl") # depends on generating positions
    include("ensembles.jl") # should be after ed.jl to reuse data
    include("lsr.jl") # should be after ed.jl to reuse data
end
