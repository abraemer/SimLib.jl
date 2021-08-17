module Ensembles

import ..ED
using ..SimLib
using XXZNumerics

using Printf: @sprintf
using Statistics: mean
using LinearAlgebra
import JLD2

export ENSEMBLE_INDICES, EnsembleDataDescriptor, EnsembleData, ensemble_predictions

## Data structure

"""
    struct EnsembleDataDescriptor

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
struct EnsembleDataDescriptor <: SimLib.AbstractDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end

EnsembleDataDescriptor(args...; kwargs...) = EnsembleDataDescriptor(ED.EDDataDescriptor(args...; kwargs...))
EnsembleDataDescriptor(edata::ED.EDData) = EnsembleDataDescriptor(descriptor(edata))

# simply forward all properties
Base.getproperty(ensdd::EnsembleDataDescriptor, p::Symbol) = p == :derivedfrom ? getfield(ensdd, :derivedfrom) : getproperty(getfield(ensdd, :derivedfrom), p)

Base.:(==)(d1::EnsembleDataDescriptor, d2::EnsembleDataDescriptor) = d1.derivedfrom == d2.derivedfrom

const ENSEMBLE_INDICES = (; microcanonical = 1, canonical = 2, diagonal = 3)

"""
    struct EnsembleData

Stores the actual data for the ensemble's predictions specified by the descriptor [`EnsembleDataDescriptor`](@ref).

# Index order
The indices mean:
 - shot number
 - field index
 - density ρ
 - ensemble
 For access use the keys :microcanonical, :canonical and :diagonal either on this struct or on ENSEMBLE_INDICES to map to the index)

The default save directory is "ensemble".
"""
struct EnsembleData <: SimLib.AbstractSimpleData
    descriptor::EnsembleDataDescriptor
    # [shot, h, rho, ensemble]
    # ensemble: 1=microcanonical, 2=canonical, 3=diag
    data::Array{Float64,4}
end

EnsembleData(args...; kwargs...) = EnsembleData(EnsembleDataDescriptor(args...; kwargs...))
EnsembleData(desc::EnsembleDataDescriptor) = EnsembleData(desc, Array{Float64,4}(undef, desc.shots, length(desc.fields), length(desc.ρs), 3))

function Base.getproperty(ensdata::EnsembleData, s::Symbol)
    if hasfield(typeof(ensdata), s)
        getfield(ensdata, s)
    elseif haskey(ENSEMBLE_INDICES, s)
        @view ensdata.data[:,:,:,ENSEMBLE_INDICES[s]]
    else 
        getproperty(ensdata.descriptor, s)
    end
end

## Saving/Loading
SimLib._filename(desc::EnsembleDataDescriptor) = filename(desc.geometry, desc.dimension, desc.system_size, desc.α)
filename(geometry, dim, N, α) = @sprintf("ensemble/%s_%id_alpha_%.1f_N_%02i", geometry, dim, α, N)

function SimLib._convert_legacy_data(::Val{:ensemble_data}, legacydata)
    data = legacydata.data
    geom = legacydata.geometry
    dim = legacydata.dim
    N = legacydata.N
    α = legacydata.α
    shots = size(data, 1)
    ρs = legacydata.ρs
    fields = legacydata.fields

    logmsg("[WARN]Unable to reconstruct parameters while loading ensembles legacy data:")
    logmsg("[WARN]  scale_fields, basis")

    savelocation = SaveLocation(prefix="")
    edd = EDDataDescriptor(geom, dim, N, α, shots, ρs, fields, missing, missing, savelocation)
    ensdd = EnsembleDataDescriptor(edd)
    EnsembleData(ensdd, data)
end

## main

# eon [eigen_state, shot, h, rho]
# eev [eigen_state, shot, h, rho]
# out              [shot, h, rho]


function _canonical!(out, eon, eev, evals)
    # out[i,j,k] = canonical_weights*eev[a,i,j,k]
    @views for I in CartesianIndices(out)
        E_0 = dot(eon[:, I], evals[:, I])
        occ = canonical_ensemble(evals[:, I], E_0) # ΔE default: 0.5% of total spectral width
        out[I] = @views dot(occ, eev[:, I])
    end
    out
end

function _microcan!(out, eon, eev, evals)
    # out[i,j,k] = microcanonical_weights*eev[a,i,j,k]
    @views for I in CartesianIndices(out)
        E_0 = dot(eon[:, I], evals[:, I])
        occ = microcanonical_ensemble(evals[:, I], E_0)
        out[I] = dot(occ, eev[:, I])
    end
    out
end

function _diagonal!(out, eon, eev, ignored=nothing)
    # out[i,j,k] = eon[a,i,j,k]*eev[a,i,j,k]
    for I in CartesianIndices(out)
        out[I] = @views dot(eon[:, I], eev[:, I])
    end
    out
end

function ensemble_predictions!(ensemble_data, eddata)
    eev = meandrop(eddata.eev; dims=1) # predict global magnetization
    eon = eddata.eon # [eigen_state, shot, h, rho]
    evals = eddata.evals # [eigen_state, shot, h, rho]

    _microcan!( @view(ensemble_data.data[:,:,:,1]), eon, eev, evals)
    _canonical!(@view(ensemble_data.data[:,:,:,2]), eon, eev, evals)
    _diagonal!( @view(ensemble_data.data[:,:,:,3]), eon, eev, evals) # evals not needed here but looks nicer ;)
    ensemble_data
end

ensemble_predictions(eddata::ED.EDData) = ensemble_predictions!(EnsembleData(EnsembleDataDescriptor(eddata)), eddata)
function SimLib.create(desc::EnsembleDataDescriptor)
    logmsg("Computing ensemble predictions for $(desc.derivedfrom)")
    eddata = load_or_create(desc.derivedfrom)
    ensemble_predictions(eddata)
end

end #module