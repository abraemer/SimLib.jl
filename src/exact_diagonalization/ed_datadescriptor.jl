## Data structure
"""
    struct EDDataDescriptor

The important bits of information needed to specify for performing exact diagonalization on disordered XXZ Heisenberg models:
 - geometry
 - dimension
 - system_size
 - α
 - shots
 - densities ρs
 - field strengths
 - scaling of field strengths
 - symmetries to respect
For `load`ing data only the first 4 fields are required
Can also be constructed from a `PositionDataDescriptor` by supplying a the missing bits (α, fields).
"""
struct EDDataDescriptor <: SimLib.AbstractDataDescriptor
    model
    diagtype::Maybe{DiagonalizationType}
    pathdata::SaveLocation
    # function EDDataDescriptor(model, diagtype, pathdata::SaveLocation)
    #     geometry ∈ SimLib.GEOMETRIES || error("Unknown geometry: $geom")
    #     scale_fields ∈ [:none, :ensemble, :shot]
    #     new(geometry, dimension, system_size, α, shots, ismissing(ρs) ? missing : unique!(sort(collect(ρs))), ismissing(fields) ? missing : unique!(sort(collect(fields))), scale_fields, basis, diagtype, pathdata::SaveLocation)
    # end
end

# _default_basis(N) = symmetrized_basis(N, Flip(N), 0)
# # unpack PositionData
# EDDataDescriptor(posdata::PositionData, args...; kwargs...) = EDDataDescriptor(descriptor(posdata), args...; kwargs...)

# # handle construction from PositionDataDescriptor and possible location overrides
# function EDDataDescriptor(posdata::PositionDataDescriptor, args...; pathdata=posdata.pathdata, prefix=pathdata.prefix, suffix=pathdata.suffix, kwargs...)
#     EDDataDescriptor(posdata.geometry, posdata.dimension, posdata.system_size, args...; ρs=posdata.ρs, shots=posdata.shots, prefix, suffix, kwargs...)
# end

# # full kwargs constructor
# function EDDataDescriptor(geometry, dimension, system_size, α; kwargs...)
#     EDDataDescriptor(; geometry, dimension, system_size, α, kwargs...)
# end

# function EDDataDescriptor(geometry, dimension, system_size; kwargs...)
#     EDDataDescriptor(; geometry, dimension, system_size, kwargs...)
# end

# # full constructor
# function EDDataDescriptor(;geometry, dimension, system_size, α, shots=missing, ρs=missing, fields=missing, scale_fields=missing, basis=_default_basis(system_size), diagtype=Full(), pathdata=SaveLocation(), prefix=pathdata.prefix, suffix=pathdata.suffix)
#     EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, diagtype, SaveLocation(prefix, suffix))
# end

# # old positional constructor
# EDDataDescriptor(posdata::PositionDataDescriptor, α, fields=missing, scale_fields=missing, basis=_default_basis(posdata.system_size), diagtype=Full(); prefix=posdata.pathdata.prefix, suffix=posdata.pathdata.suffix) =
#     EDDataDescriptor(posdata.geometry, posdata.dimension, posdata.system_size, α, posdata.shots, posdata.ρs, fields, scale_fields, basis, diagtype, SaveLocation(prefix, suffix))

# # function EDDataDescriptor(geometry, dimension, system_size, α, shots=missing, ρs=missing, fields=missing, scale_fields=missing, basis=_default_basis(system_size), diagtype=Full(), savelocation=SaveLocation(); prefix=savelocation.prefix, suffix=savelocation.suffix)
# #     EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, diagtype, SaveLocation(; prefix, suffix))
# # end

function Base.getproperty(edd::EDDataDescriptor, s::Symbol)
    if hasfield(typeof(edd), s)
        return getfield(edd, s)
    else
        return getproperty(getfield(edd, :model), s)
    end
end

function Base.:(==)(d1::EDDataDescriptor, d2::EDDataDescriptor)
    all(getfield(d1, f) == getfield(d2, f) for f in [:model, :diagtype])
end
