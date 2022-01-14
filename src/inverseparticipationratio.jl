module IPR

import ..ED
using ..SimLib
using ..SimLib: FArray
using SpinSymmetry: basissize
using SharedArrays: sdata

import Statistics

export ipr, inverse_participation_ratio, IPRData, IPRDataDescriptor, load_ipr, InverseParticipationRatio

## Data structure

"""
    struct IPRDataDescriptor <: ED.EDDerivedDataDescriptor

See [`EDDataDescriptor`](!ref) and [`EDDerivedDataDescriptor`](!ref).
"""
struct IPRDataDescriptor <: ED.EDDerivedDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end

IPRDataDescriptor(args...; kwargs...) = IPRDataDescriptor(EDDataDescriptor(args...; kwargs...))


"""
    struct IPRData

Stores the actual inverse participation ratios (IPR) specified by the descriptor [`LSRDataDescriptor`](@ref).
Inverse participation ratio is defined as IPR(|ψ⟩) = 1/(∑ᵢ |⟨i|ψ⟩|⁴ ) relative to a basis |i⟩.
Usually |i⟩ is just the z-basis and then the IPR tells a story of localization.
IPR = 2^N for maximally entangled states
IPR = 1 for maximally localized states

The first dimension hold the participation ratio of the i-th eigenstate IPR(|ϕᵢ⟩).

The default save directory is "ipr".

`Statistics.mean` and `Statistics.std` are overloaded to act on the first dimension to conveniently compute
mean LSR and its variance.
"""
struct IPRData{N} <: SimLib.AbstractSimpleData
    descriptor::IPRDataDescriptor
    data::FArray{N}
end

IPRData(iprdd::IPRDataDescriptor) = IPRData(iprdd, FArray{4}(undef, ED.ed_size(iprdd), iprdd.shots, length(iprdd.fields), length(iprdd.ρs)))

## Saving/Loading

ED._default_folder(::IPRDataDescriptor) = "ipr"

"""
    load_ipr(edd)
    load_ipr(model[, diagtype][, location])
"""
load_ipr(args...; kwargs...) = load(IPRDataDescriptor(args...; kwargs...))

## Functions

ipr(mat) = ipr!(Vector{Float64}(undef, size(mat,1)), mat)

const inverse_participation_ratio = ipr

function ipr!(v, mat)
    I, J = size(mat)
    J == length(v) || throw(ArgumentError("Sizes not compatible: $(size(v)) and $(size(mat))"))
    @inbounds for j in 1:J
        tmp = zero(Float64)
        for i in 1:I
            tmp += abs2(mat[i,j])^2
        end
        v[j] = 1/tmp
    end
    v
end

### Task

mutable struct IPRTask <: ED.EDTask
    data
end

InverseParticipationRatio() = IPRTask(nothing)

function ED.initialize!(task::IPRTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size)
end

function ED.compute_task!(task::IPRTask, evals, evecs, inds...)
    n = min(size(task.data,1), size(evecs, 2))
    @views ipr!(task.data[1:n, inds...], evecs[:, 1:n])
end

function ED.failed_task!(task::IPRTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::IPRTask, edd)
    IPRData(IPRDataDescriptor(edd), sdata(task.data))
end



end #module
