module LSR

import ..ED
using ..SimLib
using ..SimLib: FArray
using XXZNumerics: basis_size

import Statistics
using Printf: @sprintf
import JLD2

export levelspacingratio, LSRData, LSRDataDescriptor, center_region, load_lsr

## Data structure

"""
    struct LSRDataDescriptor

Carries the information to construct a [`EDDataDescriptor`](!ref) object.
 - geometry
 - dimension
 - system_size
 - α
 - shots
 - densities ρs
 - field strengths
 - scaling of field strengths
 - symmetries to respect
For `load`ing data only the first 4 fields are required.
Can also be constructed from a `PositionDataDescriptor` by supplying a the missing bits (α, fields).
"""
struct LSRDataDescriptor <: SimLib.AbstractDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end

LSRDataDescriptor(args...; kwargs...) = LSRDataDescriptor(ED.EDDataDescriptor(args...; kwargs...))
LSRDataDescriptor(edata::ED.EDData) = LSRDataDescriptor(descriptor(edata))

# simply forward all properties
Base.getproperty(lsrdd::LSRDataDescriptor, p::Symbol) = p == :derivedfrom ? getfield(lsrdd, :derivedfrom) : getproperty(getfield(lsrdd, :derivedfrom), p)

Base.:(==)(d1::LSRDataDescriptor, d2::LSRDataDescriptor) = d1.derivedfrom == d2.derivedfrom

"""
    struct LSRData

Stores the actual level-spacing ratios (LSR) specified by the descriptor [`LSRDataDescriptor`](@ref).
Level spacing ratio is defined as lsr_i = min(r_i, 1/r_i) where r_i = (E_(i-2) - E_(i-1))/(E_(i-1) - E_i)

# Index order
The indices mean:
 - i (as in lsr_i)
 - shot index
 - field index
 - density ρ

The default save directory is "lsr".

`Statistics.mean` and `Statistics.std` are overloaded to act on the first dimension to conveniently compute
mean LSR and its variance.
"""
struct LSRData <: SimLib.AbstractSimpleData
    descriptor::LSRDataDescriptor
    # [dummy, shot, h, rho]
    data::FArray{4}
end

LSRData(lsrdd::LSRDataDescriptor) = LSRData(lsrdd, FArray{4}(undef, basis_size(lsrdd.basis), lsrdd.shots, length(lsrdd.fields), length(lsrdd.ρs)))
LSRData(eddata::ED.EDData; center=1.0) = LSRData(LSRDataDescriptor(eddata), levelspacingratio(eddata.evals; center))

# forward properties to descriptor
Base.getproperty(lsr::LSRData, s::Symbol) = hasfield(LSRData, s) ? getfield(lsr, s) : getproperty(lsr.descriptor, s)


## Saving/Loading
DEFAULT_FOLDER = "lsr"

SimLib._filename(desc::LSRDataDescriptor) = filename(desc.geometry, desc.dimension, desc.system_size, desc.α)
filename(geometry, dim, N, α) = @sprintf("lsr/lsr_%s_%id_alpha_%.1f_N_%02i", geometry, dim, α, N)

load_lsr(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(LSRDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))

function SimLib._convert_legacy_data(::Val{:lsrdata}, legacydata)
    data = legacydata.data
    desc = legacydata.descriptor
    geom = desc.geometry
    dim = desc.dim
    N = desc.N
    α = desc.α
    shots = size(data, 1)
    ρs = desc.ρs
    fields = desc.fields

    logmsg("[WARN]Unable to reconstruct parameters while loading LSR legacy data:")
    logmsg("[WARN]  scale_fields, basis")

    edd = EDDataDescriptor(geom, dim, N, α, shots, ρs, fields, missing, missing, SaveLocation(prefix=""))

    LSRData(LSRDataDescriptor(edd), data)
end

function center_indices(L, center_region)
    cutoff = floor(Int, (L*(1-center_region)/2))
    (1+cutoff):L-cutoff
end

center_region(lsr::LSRData, center) = @view lsr.data[center_indices(size(lsr.data, 1), center), :, :, :]

levelspacingratio(eddata::ED.EDData; center=1.0) = LSRData(eddata; center)

function levelspacingratio(levels; center=1.0)
    sizes = size(levels)
    range = center_indices(sizes[1]-2, center) .+ 2
    res = Array{Float64, length(sizes)}(undef, length(range), sizes[2:end]...)
    for I in CartesianIndices(axes(levels)[2:end])
        for (i,j) in enumerate(range)
            a,b,c = levels[j-2,I], levels[j-1,I], levels[j,I]
            if a ≈ b || b ≈ c || a ≈ c
                # prevent NaNs or -Infs
                # a=b=c -> NaN
                # a>b=c -> -Inf # this can happen if all of them are very close
                res[i, I] = 0
            else
                ratio = abs((b-a)/(c-b)) # use abs to be sure
                res[i, I] = min(ratio, 1/ratio)
            end
        end
    end
    res
end

function SimLib.create(lsrdd::LSRDataDescriptor)
    eddata = SimLib.load_or_create(lsrdd.derivedfrom)
    LSRData(eddata)
end

Statistics.mean(lsr::LSRData; center=1.0) = meandrop(center_region(lsr, center); dims=1)
Statistics.std(lsr::LSRData; center=1.0) = stddrop(center_region(lsr, center); dims=1)
end #module
