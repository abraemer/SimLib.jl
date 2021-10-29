module ED

using ..Positions
using ..SimLib
using ..SimLib: Maybe, FArray

using Arpack
using Distributed
import JLD2
using LinearAlgebra
using Printf: @sprintf
using SharedArrays
using XXZNumerics
using SpinSymmetry

export EDDataDescriptor, Full, Sparse, ed_size, run_ed

include("ed_task.jl")
include("ed_core.jl")

end #module
