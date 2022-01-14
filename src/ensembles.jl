module Ensembles

import ..ED
using ..Levels
using ..OPDiag
using ..EON
using ..SimLib
using XXZNumerics

using Printf: @sprintf
using Statistics: mean
using LinearAlgebra
import JLD2

export ENSEMBLE_INDICES, EnsembleDataDescriptor, EnsembleData, ensemble_predictions, load_ensemble

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
struct EnsembleDataDescriptor <: ED.EDDerivedDataDescriptor
    derivedfrom::ED.EDDataDescriptor
    EnsembleDataDescriptor(args...; kwargs...) = new(EDDataDescriptor(args...; kwargs...))
end

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
struct EnsembleData{N} <: SimLib.AbstractSimpleData
    descriptor::EnsembleDataDescriptor
    # [shot, h, rho, ensemble]
    # ensemble: 1=microcanonical, 2=canonical, 3=diag
    data::Array{Float64,N}
end

_data_indices(arr) = CartesianIndices(axes(arr)[1:end-1])

function Base.getproperty(ensdata::EnsembleData, s::Symbol)
    if hasfield(typeof(ensdata), s)
        getfield(ensdata, s)
    elseif haskey(ENSEMBLE_INDICES, s)
        @view ensdata.data[_data_indices(ensdata.data), ENSEMBLE_INDICES[s]]
    else
        getproperty(ensdata.descriptor, s)
    end
end

## Saving/Loading
ED._default_folder(::EnsembleDataDescriptor) = "ensemble"

load_ensemble(args...; kwargs...) = load(EnsembleDataDescriptor(args...; kwargs...))
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
        if E_0 < evals[1,I] || E_0 > evals[end,I] || !issorted(evals[1,I])
            logmsg("Something off for index $I")
            @show issorted(evals[:,I])
            @show norm(sort(evals[:, I]) - evals[:, I])
            @show E_0, evals[1,I], evals[end,I], minimum(evals[:,I]), maximum(evals[:,I])
            @show eltype(eon)
            @show sum(eon[:, I])
            println()
        end
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

function ensemble_predictions(evals, eon, eev)
    ensemble_data = zeros(Float64, size(eon)[2:end]..., 3)
    ensemble_predictions!(ensemble_data, evals, eon, eev)
    return ensemble_data
end

function ensemble_predictions!(ensemble_data, evals, eon, eev)
    if !(eltype(eon) <: Real)
        eon = abs2.(eon)
        logmsg("Ensemble prediction: Got amplitudes -> squaring.")
    end
    I = _data_indices(evals)
    _microcan!( @view(ensemble_data[I, 1]), eon, eev, evals)
    _canonical!(@view(ensemble_data[I, 2]), eon, eev, evals)
    _diagonal!( @view(ensemble_data[I, 3]), eon, eev, evals) # evals not needed here but looks nicer ;)
end

function ensemble_predictions(levels::LevelData, eon::EONData, eev::OPDiagData)
    data = ensemble_predictions(levels.data, eon.data, eev.data)
    return EnsembleData(EnsembleDataDescriptor(descriptor(levels).derivedfrom), data)
end

function SimLib.create(desc::EnsembleDataDescriptor)
    eddesc = desc.derivedfrom
    logmsg("Computing ensemble predictions for $(eddesc)")
    levels = load(LevelDataDescriptor(eddesc))
    eon = load(EONDataDescriptor("xpol", eddesc))
    eev = load(OPDiagDataDescriptor("xmag", eddesc))
    return ensemble_predictions(levels, eon, eev)
end

end #module
