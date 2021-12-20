module SimLib

using Dates
import JLD2
using LinearAlgebra
using Reexport
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
include("exact_diagonalization/exact_diagonalization.jl")
include("eigenstate_occupation.jl")
include("eigenstate_locality.jl")
include("levels.jl")
include("operator_diagonal.jl")
include("ensembles.jl")
include("levelspacingratio.jl")
include("halfchainentropy_zblock.jl")
include("inverseparticipationratio.jl")

@reexport using .Positions
@reexport using .ED
@reexport using .Levels
@reexport using .EON
@reexport using .OPDiag
@reexport using .LSR
@reexport using .EL
@reexport using .Ensembles
@reexport using .HCE_ZBlock_Module
@reexport using .IPR

end
