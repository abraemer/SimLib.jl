module ED

using ..Positions
using ..SimLib
using ..SimLib: Maybe, FArray

using Distributed
import JLD2
using LinearAlgebra
using Printf: @sprintf
using SharedArrays
using TimerOutputs
using XXZNumerics
using SpinSymmetry

export EDDataDescriptor, run_ed

get_stats() = TimerOutputs.get_timer("ED")
reset_stats() = reset_timer!(get_stats())
show_stats() = print_timer(IOContext(stdout, :limit=>false, :displaysize=>(100,400)), get_stats())

include("ed_task.jl")
include("ed_core.jl")

end #module
