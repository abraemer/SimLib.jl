module Positions

import JLD2
using Printf: @sprintf
using SimLib
using XXZNumerics

export PositionDataDescriptor, PositionData, position_datapath, save, load, create_positions!, load_or_create

## Data structure

"""
    struct PositionDataDescriptor

The important bits of information needed to specify a set positions are:
 - geometry
 - dimension
 - system_size
 - shots
 - densities ρs
"""
struct PositionDataDescriptor
    geometry::Symbol
    dimension::Int
    system_size::Int
    shots::Int
    ρs::Vector{Float64}
    pathdata::SaveLocation
    function PositionDataDescriptor(geom, dimension, system_size, shots, ρs, pathdata::SaveLocation)
        geom ∈ SimLib.GEOMETRIES || error("Unknown geometry: $geom")
        new(geom, dimension, system_size, shots, sort(vec(ρs)), pathdata)
    end
end

function PositionDataDescriptor(geom, dimension, system_size, shots, ρs, prefix=path_prefix(), suffix="")
    PositionDataDescriptor(geom, dimension, system_size, shots, ρs, SaveLocation(prefix, suffix))
end

function Base.:(==)(d1::PositionDataDescriptor, d2::PositionDataDescriptor)
    all(getfield(d1, f) == getfield(d2, f) for f in [:geometry, :dimension, :system_size, :shots, :ρs])
end

Base.string(desc::PositionDataDescriptor) = @sprintf("%s_%id_N_%02i", desc.geometry, desc.dimension, desc.system_size)

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
struct PositionData
    descriptor::PositionDataDescriptor
    # [xyz, N, shot, ρ]
    data::Array{Float64, 4}
end

# forward properties to descriptor
Base.getproperty(posdata::PositionData, s::Symbol) = hasfield(PositionData, s) ? getfield(posdata, s) : getproperty(posdata.descriptor, s)

PositionData(desc::PositionDataDescriptor) = PositionData(desc, zeros(Float64, desc.dimension, desc.system_size, desc.shots, length(desc.ρs)))
PositionData(geom, ρs, shots, dimension, prefix=path_prefix(), suffix="") = PositionData(PositionDataDescriptor(geom, ρs, shots, dimension, SaveLocation(prefix, suffix)))

Base.getindex(posdata::PositionData, inds...) = getindex(posdata.data, inds...)
Base.setindex!(posdata::PositionData, args...) = setindex!(posdata.data, args...)

## Saving/Loading
# slowly destructure the input
position_datapath(posdata::PositionData, args...) = position_datapath(posdata.descriptor, args...)
position_datapath(desc::PositionDataDescriptor) = position_datapath(desc, desc.pathdata)
position_datapath(desc::PositionDataDescriptor, pathdata::SaveLocation) = position_datapath(desc, pathdata.prefix, pathdata.suffix)
function position_datapath(desc::PositionDataDescriptor, prefix::AbstractString, suffix::AbstractString="") 
    if length(suffix) > 0
        joinpath(prefix, "positions", "$(string(desc))-$(suffix).jld2")
    else
        joinpath(prefix, "positions", "$(string(desc)).jld2")
    end
end

save(posdata::PositionData, args...) = save(posdata, position_datapath(posdata, args...))

function save(posdata::PositionData, path::AbstractString)
    dname = dirname(path)
    if !isdir(dname)
        logmsg("Save directory: $dname does not exists. Creating!")
        mkpath(dname)
    end
    logmsg("Saving file: $path")
    JLD2.jldsave(path; posdata)
end

load(desc::PositionDataDescriptor, args...) = load(position_datapath(desc, args...))
function load(path::AbstractString) 
    isfile(path) || error("$(abspath(path)) does not exist!")
    JLD2.load(path)["posdata"]
end

#load(prefix, geometry, N, dim) = load(position_datapath(prefix, geometry, N, dim))

## main function
function create_positions!(empty_posdata; fail_rate=0.3)
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

function load_or_create(desc::PositionDataDescriptor, pathargs...; save=true)
    p = position_datapath(desc, pathargs...)
    if !isfile(p)
        logmsg("No position data found!")
        _createAndSave(desc, p, save)
    else
        logmsg("Found existing position data!")
        data = load(p)
        comp = false
        ## this try catch is for the case that an invalid is loaded and thus not all attributes may be present
        ## f.e. if the file was written by a previous version of this code
        try
            comp = data.desc == desc
        catch e
            logmsg(e)
        end
        if comp
            data
        else
            logmsg("Loaded data does not fit requirements: generating anew.")
            _createAndSave(desc, p, save)
        end
    end
end

function _createAndSave(desc, path, dosave)
    data = PositionData(desc)
    logmsg("Creating Position data with $(desc.shots) shots, $(desc.geometry) N=$(desc.system_size) $(desc.dimension)d and $(length(desc.ρs)) densities")
    create_positions!(data)
    if dosave
        logmsg("Saving newly created position data for next time!")
        save(data, path)
    end
    data
end

end #module