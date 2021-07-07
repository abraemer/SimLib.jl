module Positions

using JLD2
using Printf: @sprintf
using SimLib
using XXZNumerics

export PositionData, dimension, system_size, shots, ρs, geometry, data, position_datapath, save, load, create_positions!

## Data structure

struct PositionData
    geometry::Symbol
    ρs::Vector{Float64}
    coords::Array{Float64, 4}
end

PositionData(geometry, ρs, shots, system_size, dim) = PositionData(geometry, vec(ρs), zeros(Float64, dim, system_size, shots, length(ρs)))

dimension(posdata::PositionData) = size(posdata.coords, 1)
system_size(posdata::PositionData) = size(posdata.coords, 2)
shots(posdata::PositionData) = size(posdata.coords, 3)
ρs(posdata::PositionData) = posdata.ρs
geometry(posdata::PositionData) = posdata.geometry
data(posdata::PositionData) = posdata.coords

Base.getindex(posdata::PositionData, inds...) = getindex(data(posdata), inds...)
Base.setindex!(posdata::PositionData, args...) = setindex!(data(posdata), args...)

## Saving/Loading
position_datapath(prefix, geometry, N, dim) = joinpath(prefix, "positions", @sprintf("%s_%id_N_%02i.jld2", geometry, dim, N))
position_datapath(prefix, posdata::PositionData) = position_datapath(prefix, geometry(posdata), system_size(posdata), dimension(posdata))

function save(prefix, posdata::PositionData)
    path = position_datapath(prefix, posdata)
    mkpath(dirname(path))
    JLD2.jldsave(path; posdata)
end

load(path) = JLD2.load(path)["posdata"]
load(prefix, geometry, N, dim) = load(position_datapath(prefix, geometry, N, dim))

## main function
function create_positions!(empty_posdata; fail_rate=0.3)
    dim = dimension(empty_posdata)
    N = system_size(empty_posdata)

    max_misses = shots(empty_posdata)*fail_rate
    for (i, ρ) in enumerate(ρs(empty_posdata))
        sampler = geometry_from_density(geometry(empty_posdata), ρ, N, dim)
        misses = 0
        for j in 1:shots(empty_posdata)
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

end #module