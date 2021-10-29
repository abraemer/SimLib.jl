module HCE_ZBlock_Module

import ..ED
using ..SimLib
using ..SimLib: FArray, Maybe
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



HCEDataDescriptor(L, symm, args...; kwargs...) = HCEDataDescriptor(L, symm, EDDataDescriptor(args...; kwargs...))



### Data obj

struct HCEData <: ED.EDDerivedData
    descriptor::HCEDataDescriptor
    data::FArray{5}
end

ED._default_folder(::HCEDataDescriptor) = "entropy"
ED._filename_addition(hcedd::HCEDataDescriptor) = "-l_$(hcedd.L)" * (hcedd.symm ? "_symm" : "")

load_entropy(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(HCEDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))


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

mutable struct HalfChainEntropyTask <: ED.EDTask
    L::Maybe{Int}
    symm::Bool
    entropy_strategy
    data
end

## ToDo: This always symmetrizes over the chain right now.

HalfChainEntropyZBlock(L=missing) = HalfChainEntropyTask(L, true, nothing, nothing)
HalfChainEntropyZBlock(; L=missing) = HalfChainEntropyZBlock(L)

function ED.initialize!(task::HalfChainEntropyTask, edd, arrayconstructor)
    if ismissing(task.L)
        ## TODO this is not universal! basis might also be a ZBlockBasis directly...
        task.L = div(edd.basis.basis.k, 2) # half-chain is default
    end
    task.entropy_strategy = SymmZBlockEntanglementEntropy(edd.basis.basis, task.L)
    task.data = arrayconstructor(Float64, task.entropy_strategy.size, ED.ed_size(edd), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::HalfChainEntropyTask, ρindex, shot, fieldindex, evals, evecs)
    for (i, ψ) in enumerate(eachcol(evecs))
        entanglement_entropy!(view(task.data, :, i, shot, fieldindex, ρindex), task.entropy_strategy, ψ)
    end
end

function ED.failed_task!(task::HalfChainEntropyTask, ρindex, shot, fieldindex)
    task.data[:, :, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::HalfChainEntropyTask, edd)
    HCEData(HCEDataDescriptor(task.L, task.symm, edd), sdata(task.data))
end

Base.summary(task::HalfChainEntropyTask) = string(typeof(task)) * "(L=$(task.L), symm=$(task.symm))"

end # module
