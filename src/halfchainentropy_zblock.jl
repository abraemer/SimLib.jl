module HCE_ZBlock_Module

import ..ED
using ..SimLib
using ..SimLib: FArray, Maybe
using LinearAlgebra: eigvals!, Hermitian, mul!
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
_filename_addition(hcedd::HCEDataDescriptor) = "-l_$(hcedd.L)" * (hcedd.symm ? "_symm" : "")

load_entropy(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(HCEDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))


function entanglement_entropy_zblock(state, N, kB, N1)
	(N == N1 || N1 == 0) && return 0
	(N == k  ||  k == 0) && return 0
	reducedρ = []
	for k in reverse(0:min(N1,kB))
		let b1 = SpinSymmetry._indices(ZBlockBasis(N1,k)),
			b2 = 2^N1 * (SpinSymmetry._indices(ZBlockBasis(NB-N1, kB-k)) .- 1),
			temp = zeros(eltype(ψsymm), length(b2), length(b2))
			for indA in b1
				fullBasisInds = indA .+ b2
				symmBasisInds = indexLookup[fullBasisInds]
				vals = state[symmBasisInds]
				temp += vals * vals'
			end
			push!(reducedρ, temp)
		end
	end
	entropy(vcat(eigvals!.(reducedρ)...))
end

_zblock_inds(N, k) = SpinSymmetry._indices(zbasis(N, k))

struct SymmZBlockEntanglementEntropy
    Nfull::Int
    kfull::Int
    N1::Int
    indexLookup::Vector{Int}## ToDo: is that a sensible data structure?
    allinds::Vector{Matrix{Int64}}
    #krange::UnitRange{Int}
    size::Int
    function SymmZBlockEntanglementEntropy(zblockbasis, N1)
        N = zblockbasis.N
        k = zblockbasis.k
        N1 = max(N1, N-N1)
        #inds = SpinSymmetry._indices(zblockbasis)
        #indexLookup = Dict((i, pos) for (pos, i) in enumerate(inds))
        indexLookup = zeros(2^N)
        indexLookup[_zblock_inds(N, k)] = 1:binomial(N,k)
        krange = max(0,N1+k-N):min(N1, k)
        allinds = [2^N1 .* (_zblock_inds(N-N1, k-ki) .- 1) .+ _zblock_inds(N1, ki)' for ki in krange]
        size = 2*N1 == N ? N1 : N
        new(N, k, N1, indexLookup, allinds, size)
    end
end

# scalar for broadcasting
Base.broadcastable(s::SymmZBlockEntanglementEntropy) = Ref(s)

entanglement_entropy(s::SymmZBlockEntanglementEntropy, ψ) = entanglement_entropy!(zeros(Float64, s.size),s,ψ)

function entanglement_entropy!(out, s::SymmZBlockEntanglementEntropy, ψ)
    NB = s.Nfull
    for indmat in s.allinds
        l = size(indmat, 1)
		temp = zeros(eltype(ψ), l, l)
		vals = zeros(eltype(ψ), l)
        for shift in 1:s.size
            fill!(temp, 0)
            for inds in eachcol(indmat)
                #vals .= getindex.(Ref(ψ), getindex.(Ref(s.indexLookup), SpinSymmetry._roll_bits.(NB, inds .- 1, shift-1) .+ 1))
                for i in 1:l
                    vals[i] = ψ[s.indexLookup[SpinSymmetry._roll_bits(NB, inds[i]-1, shift-1)+1]]
                end
                #temp += vals * vals'
                temp = mul!(temp, vals, vals', 1, 1) # compute temp += vals * vals'
            end
            #println("diag $(size(temp))")
            out[shift] += entropy(eigvals!(Hermitian(temp)))
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
    task.data = arrayconstructor(Float64, task.entropy_strategy.size, basissize(edd.basis), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::HalfChainEntropyTask, ρindex, shot, fieldindex, eigen)
    for (i, ψ) in enumerate(eachcol(eigen.vectors))
        entanglement_entropy!(view(task.data, :, i, shot, fieldindex, ρindex), task.entropy_strategy, ψ)
    end
end

function ED.failed_task!(task::HalfChainEntropyTask, ρindex, shot, fieldindex)
    task.data[:, :, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::HalfChainEntropyTask, edd)
    HCEData(HCEDataDescriptor(task.L, task.symm, edd), sdata(task.data))
end

end # module
