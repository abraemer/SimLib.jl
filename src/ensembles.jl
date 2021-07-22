module Ensembles

import ..ED
using XXZNumerics

using Printf: @sprintf
using Statistics: mean
using LinearAlgebra
import JLD2

## Data structure

# index order
# shot, h, rho, ensemble
struct EnsembleData
    geometry::Symbol
    dim::Int64
    N::Int64
    α::Float64
    ρs::Vector{Float64}
    fields::Vector{Float64}
    # [shot, h, rho, ensemble]
    # ensemble: 1=microcanonical, 2=canonical, 3=diag
    data::Array{Float64,4} 
    function EnsembleData(geometry, dim, α, ρs, shots, system_size, fields)
        new(geometry,
            dim,
            system_size,
            α,
            ρs,
            fields,
            Array{Float64,4}(undef, shots, length(fields), length(ρs), 3))
    end
end

EnsembleData(eddata::ED.EDData) = EnsembleData(ED.geometry(eddata), ED.dimension(eddata), ED.α(eddata), ED.ρ_values(eddata), ED.shots(eddata), ED.system_size(eddata), ED.fields(eddata))


system_size(data::EnsembleData) = data.N
shots(data::EnsembleData) = size(data.data, 1)
ρ_values(data::EnsembleData) = data.ρs
fields(data::EnsembleData) = data.fields
data(data::EnsembleData) = data.data
α(data::EnsembleData) = data.α
geometry(data::EnsembleData) = data.geometry
dimension(data::EnsembleData) = data.dim

## Saving/Loading
ensemble_datapath(prefix, geometry, dim, N, α) = joinpath(prefix, "ensemble", @sprintf("%s_%id_alpha_%.1f_N_%02i.jld2", geometry, dim, α, N))
ensemble_datapath(prefix, data::EnsembleData) = ensemble_datapath(prefix, geometry(data), dimension(data), system_size(data), α(data))

function save(prefix, ensemble_data::EnsembleData)
    path = ensemble_datapath(prefix, ensemble_data)
    mkpath(dirname(path))
    JLD2.jldsave(path; ensemble_data)
end

load(path) = JLD2.load(path)["ensemble_data"]
load(prefix, geometry, N, dim, α) = load(ensemble_datapath(prefix, geometry, dim, N, α))

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
    eev = dropdims(mean(ED.eev(eddata); dims=1); dims=1) # predict global magnetization
    eon = ED.eon(eddata) # [eigen_state, shot, h, rho]
    evals = ED.evals(eddata) # [eigen_state, shot, h, rho]

    _microcan!( @view(ensemble_data.data[:,:,:,1]), eon, eev, evals)
    _canonical!(@view(ensemble_data.data[:,:,:,2]), eon, eev, evals)
    _diagonal!( @view(ensemble_data.data[:,:,:,3]), eon, eev, evals) # evals not needed here but looks nicer ;)
    ensemble_data
end

ensemble_predictions(eddata::ED.EDData) = ensemble_predictions!(EnsembleData(eddata), eddata)

end #module