## Data structure
"""
    struct EDDataDescriptor

    EDDataDescriptor(model[, diagtype][, pathdata]; suffix, prefix)

The important bits of information needed to specify for performing exact diagonalization on spin models:
 - `model`: See [`EDModels`](@ref)
 - `diagtype`: See [`DiagonalizationType`](@ref)
 - `pathdata`: `SaveLocation` for the results
For `load`ing data only the first field is required!

Note: This datastructure forwards properties of `model`.
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
    function EDDataDescriptor(model, diagtype=missing, pathdata=SaveLocation(); suffix=pathdata.suffix, prefix=pathdata.prefix)
        new(model, diagtype, SaveLocation(;suffix, prefix))
    end
end

# constructor for easier loading
EDDataDescriptor(model, pathdata::SaveLocation=SaveLocation(); kwargs...) = EDDataDescriptor(model, missing, pathdata; kwargs...)

# forward properties to model
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
