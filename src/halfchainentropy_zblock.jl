module HCE_ZBlock_Module

import ..ED
using ..SimLib
using ..SimLib: FArray
using LinearAlgebra: eigvals!, Hermitian, mul!, svdvals!
using SharedArrays: sdata
using SpinSymmetry
using XXZNumerics: entropy


export HalfChainEntropyZBlock, HCEDataDescriptor, HCEData, load_entropy

### Descriptor

struct HCEDataDescriptor <: ED.EDDerivedDataDescriptor
    L::Int
    symm::Bool
    derivedfrom::ED.EDDataDescriptor
end

HCEDataDescriptor(L, args...; kwargs...) = HCEDataDescriptor(L, true, EDDataDescriptor(args...; kwargs...))
HCEDataDescriptor(L, symm::Bool, args...; kwargs...) = HCEDataDescriptor(L, symm, EDDataDescriptor(args...; kwargs...))



### Data obj

struct HCEData{N} <: SimLib.AbstractSimpleData
    descriptor::HCEDataDescriptor
    data::FArray{N}
end

ED._default_folder(::HCEDataDescriptor) = "entropy"
ED._filename_addition(hcedd::HCEDataDescriptor) = "-l_$(hcedd.L)" * (hcedd.symm ? "_symm" : "")

"""
    load_entropy(L[, symm], edd)
    load_entropy(L[, symm], model[, diagtype][, location])
"""

load_entropy(args...; kwargs...) =  load(HCEDataDescriptor(args...; kwargs...))


## Functions

_zblock_inds(N, k) = SpinSymmetry._indices(zbasis(N, k))

struct SymmZBlockEntanglementEntropy
    Nfull::Int
    kfull::Int
    N1::Int
    indexLookup::Vector{Int}## ToDo: is that a sensible data structure?
    indsA::Vector{Vector{Int}} # note that indsA contains fewer indices -> should be used as column
    indsB::Vector{Vector{Int}}
    size::Int
    function SymmZBlockEntanglementEntropy(zblockbasis, N1)
        N = zblockbasis.N
        k = zblockbasis.k
        N1 = max(N1, N-N1)
        indexLookup = zeros(2^N)
        indexLookup[_zblock_inds(N, k)] = 1:binomial(N,k)
        krange = max(0,N1+k-N):min(N1, k)
        indsA = [2^N1 .* (_zblock_inds(N-N1, k-ki) .- 1) for ki in krange]
        indsB = [_zblock_inds(N1, ki) .- 1 for ki in krange]
        size = 2*N1 == N ? N1 : N
        new(N, k, N1, indexLookup, indsA, indsB, size)
    end
end

# scalar for broadcasting
Base.broadcastable(s::SymmZBlockEntanglementEntropy) = Ref(s)

entanglement_entropy(s::SymmZBlockEntanglementEntropy, ψ) = entanglement_entropy!(zeros(Float64, s.size),s,ψ)

function entanglement_entropy!(out, s::SymmZBlockEntanglementEntropy, ψ)
    NB = s.Nfull
    fill!(out, 0)
    for (indA, indB) in zip(s.indsA, s.indsB)
        ## No point in optimizing this allocation further.
        mat = Matrix{eltype(ψ)}(undef, length(indA), length(indB))
        for shift in 1:s.size
            mat .= getindex.(Ref(ψ), getindex.(Ref(s.indexLookup), SpinSymmetry._roll_bits.(NB, indA .+ indB', shift-1) .+ 1))
            out[shift] += entropy(svdvals!(mat) .^ 2) #  most allocations come from svdvals!
		end
	end
    out
end

### Task

mutable struct HalfChainEntropyTask{S} <: ED.EDTask
    L::Int
    symm::Bool
    entropy_strategy::S
    data
end

## ToDo: This always symmetrizes over the chain right now.

HalfChainEntropyZBlock(basis::SymmetrizedBasis, L=div(basis.basis.N,2)) = HalfChainEntropyTask(L, true, SymmZBlockEntanglementEntropy(basis.basis, L), nothing)
HalfChainEntropyZBlock(; basis, L=div(basis.basis.N,2)) = HalfChainEntropyZBlock(basis, L)


function ED.initialize!(task::HalfChainEntropyTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, task.entropy_strategy.size, spectral_size)
end

function ED.compute_task!(task::HalfChainEntropyTask, evals, evecs, inds...)
    for (i, ψ) in enumerate(eachcol(evecs))
        entanglement_entropy!(view(task.data, :, i, inds...), task.entropy_strategy, ψ)
    end
end

function ED.failed_task!(task::HalfChainEntropyTask, inds...)
    task.data[:, :, inds...] .= NaN64
end

function ED.assemble(task::HalfChainEntropyTask, edd)
    HCEData(HCEDataDescriptor(task.L, task.symm, edd), sdata(task.data))
end

Base.summary(task::HalfChainEntropyTask) = string(typeof(task)) * "(L=$(task.L), symm=$(task.symm))"

end # module
