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
    struct IPRDataDescriptor

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
struct IPRDataDescriptor <: ED.EDDerivedDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end

IPRDataDescriptor(args...; kwargs...) = IPRataDescriptor(EDDataDescriptor(args...; kwargs...))


"""
    struct IPRData

Stores the actual inverse participation ratios (IPR) specified by the descriptor [`LSRDataDescriptor`](@ref).
Inverse participation ratio is defined as IPR(|ψ⟩) = 1/(∑ᵢ |⟨i|ψ⟩|⁴ ) relative to a basis |i⟩.
Usually |i⟩ is just the z-basis and then the IPR tells a story of localization.
IPR = 2^N for maximally entangled states
IPR = 1 for maximally localized states

# Index order
The indices mean:
 - i (as in IPR(|ϕᵢ⟩))
 - shot index
 - field index
 - density ρ

The default save directory is "ipr".

`Statistics.mean` and `Statistics.std` are overloaded to act on the first dimension to conveniently compute
mean LSR and its variance.
"""
struct IPRData <: ED.EDDerivedData
    descriptor::IPRDataDescriptor
    # [state, shot, h, rho]
    data::FArray{4}
end

IPRData(lsrdd::IPRDataDescriptor) = IPRData(lsrdd, FArray{4}(undef, basissize(lsrdd.basis), lsrdd.shots, length(lsrdd.fields), length(lsrdd.ρs)))

## Saving/Loading

ED._default_folder(::IPRDataDescriptor) = "ipr"

load_ipr(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(IPRDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))

## Functions

ipr(mat) = ipr!(Vector{Float64}(undef, size(mat,1)), mat)

const inverse_participation_ratio = ipr

function ipr!(v, mat)
    I, J = size(mat)
    @inbounds for j in 1:J
        tmp = zero(Float64)
        for i in I
            tmp += abs2(mat[i,j])^2
        end
        v[j] = tmp
    end
    v
end

### Task

mutable struct IPRTask <: ED.EDTask
    data
end

InverseParticipationRatio() = IPRTask(nothing)

function ED.initialize!(task::IPRTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, basissize(edd.basis), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::IPRTask, ρindex, shot, fieldindex, eigen)
    ipr!(view(task.data, :,shot, fieldindex, ρindex), eigen.vectors)
end

function ED.failed_task!(task::IPRTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::IPRTask, edd)
    IPRData(IPRDataDescriptor(edd), sdata(task.data))
end



end #module
