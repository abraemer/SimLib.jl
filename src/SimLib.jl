module SimLib

using Dates
import JLD2
using LinearAlgebra
using Statistics
using XXZNumerics


export logmsg, path_prefix, parse_geometry, geometry_from_density, meandrop, stddrop
export SaveLocation, descriptor, datapath, create, save, load, load_or_create

# simplify type definitions
const FArray{N} = Array{Float64, N} where N
const Maybe{T} = Union{Missing, T} where T

include("general.jl")
include("data.jl")
include("positions.jl")
include("ed.jl")
include("ensembles.jl")
include("lsr.jl")

using .Positions
export Positions, PositionDataDescriptor, PositionData

using .ED
export ED, EDDataDescriptor, EDData, run_ed

using .Ensembles
export Ensembles, EnsembleDataDescriptor, EnsembleData, ENSEMBLE_INDICES, ensemble_predictions

using .LSR
export LSR, LSRDataDescriptor, LSRData, levelspacingratio
end
