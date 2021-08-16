module Positions

import JLD2
using Printf: @sprintf
using ..SimLib
using ..SimLib: Maybe, FArray
using XXZNumerics

export PositionDataDescriptor, PositionData

## Data structure

"""
    struct PositionDataDescriptor

The important bits of information needed to specify a set positions are:
 - geometry
 - dimension
 - system_size
 - shots
 - densities ρs
For `load`ing data the last 2 may be omitted.
"""
struct PositionDataDescriptor <: SimLib.AbstractDataDescriptor
    geometry::Symbol
    dimension::Int
    system_size::Int
    shots::Maybe{Int}
    ρs::Maybe{FArray{1}}
    pathdata::SaveLocation
    function PositionDataDescriptor(geom, dimension, system_size, shots, ρs, pathdata::SaveLocation)
        geom ∈ SimLib.GEOMETRIES || error("Unknown geometry: $geom")
        new(geom, dimension, system_size, shots, ismissing(ρs) ? missing : unique!(sort(vec(ρs))), pathdata)
    end
end

function PositionDataDescriptor(geom, dimension, system_size, shots=missing, ρs=missing; prefix=path_prefix(), suffix="")
    PositionDataDescriptor(geom, dimension, system_size, shots, ρs, SaveLocation(prefix, suffix))
end

function Base.:(==)(d1::PositionDataDescriptor, d2::PositionDataDescriptor)
    all(getfield(d1, f) == getfield(d2, f) for f in [:geometry, :dimension, :system_size, :shots, :ρs])
end

SimLib._filename(desc::PositionDataDescriptor) = filename(desc.geometry, desc.dimension, desc.system_size)
filename(geometry, dimension, system_size) = @sprintf("positions/%s_%id_N_%02i", geometry, dimension, system_size)

"""
    struct PositionData

Stores the actual data for the positions specified by the descriptor [`PositionDataDescriptor`](@ref).
The indices mean:
 - coordinate index
 - particle
 - shot number
 - density ρ
Thus the data as dimensions: [D, N, #SHOT, #ρ]

The default save directory is "positions".
"""
struct PositionData <: SimLib.AbstractSimpleData
    descriptor::PositionDataDescriptor
    # [xyz, N, shot, ρ]
    data::Array{Float64, 4}
end


PositionData(desc::PositionDataDescriptor) = PositionData(desc, zeros(Float64, desc.dimension, desc.system_size, desc.shots, length(desc.ρs)))
PositionData(args...; kwargs...) = PositionData(PositionDataDescriptor(args..., kwargs...))

function SimLib._convert_legacy_data(::Val{:posdata}, legacydata)
    data = legacydata.coords
    dim, N, shots, _ = size(data)
    ρs = legacydata.ρs
    geom = legacydata.geometry

    savelocation = SaveLocation(prefix="", suffix="")
    desc = PositionDataDescriptor(geom, dim, N, shots, ρs, savelocation)

    PositionData(desc, data)
end

## main function
function create_positions!(empty_posdata; fail_rate=0.3)
    logmsg("Generating positions for $(empty_posdata.descriptor)")
    dim = empty_posdata.dimension
    N = empty_posdata.system_size
    shots = empty_posdata.shots

    max_misses = shots*fail_rate
    for (i, ρ) in enumerate(empty_posdata.ρs)
        sampler = geometry_from_density(empty_posdata.geometry, ρ, N, dim)
        misses = 0
        for j in 1:shots
            positions = Vector{Vector{Float64}}()
            while length(positions) == 0 && misses < max_misses
                try
                    positions = sample_blockaded(sampler, N)
                catch e;
                    misses += 1
                    if !(e isa ErrorException)
                        logmsg("Got unexpected Error: $e\nCurrent ρ=$ρ, dim=$dim, shot=$j, N=$N")
                    end
                end
            end
            if misses >= max_misses
                error("sampler $sampler did fail to converge too often (fail rate $(round(fail_rate*100; digits=1))%! Current ρ=$ρ, dim=$dim, shot=$j, N=$N")
            end
            empty_posdata[:,:,j,i] = hcat(positions...)
        end
    end
    empty_posdata # now filled
end

SimLib.create(desc::PositionDataDescriptor) = create_positions!(PositionData(desc))

end #module