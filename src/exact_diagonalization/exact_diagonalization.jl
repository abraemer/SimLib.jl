module ED

using ..Positions
using ..SimLib
using ..SimLib: Maybe, FArray

using Arpack
using Distributed
using FilteredMatrices
import JLD2
using KrylovKit
using LinearAlgebra
using LinearMaps
using Printf: @sprintf
using Reexport
using SharedArrays
using XXZNumerics
using SpinSymmetry

export EDDataDescriptor, EDDerivedDataDescriptor, EDDerivedData, Full, Sparse, ShiftInvert, ShiftInvertARPACK, POLFED, ed_size, run_ed
export Serial, Threaded, Parallel

include("task.jl")
include("diagonalizationtype.jl")
include("ed_datadescriptor.jl")
include("ed_derived_descriptor.jl")
include("runmode.jl")
include("models/models.jl")
include("ed_core.jl")

@reexport using .EDModels

end #module
