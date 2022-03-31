module PRModule

import ..ED
using .. SimLib
using ..SimLib: FArray
using LinearAlgebra: mul!, I
using SharedArrays: sdata
using SparseArrays: SparseMatrixCSC, sparse
using SpinSymmetry

export pr, participation_ratio, ParticipationRatio, PRDataDescriptor, PRData, load_pr
export ZBasis, PairBasis, PetersMethod, NaivePairing

function participation_ratio(vec::AbstractVector)
    return 1/sum(abs2∘abs2, vec)
end

participation_ratio(mat::AbstractMatrix) = participation_ratio!(zeros(float(eltype(mat)), size(mat,2)), mat)

function participation_ratio!(out, mat::AbstractMatrix)
    out .= participation_ratio.(eachcol(mat))
end

const pr = participation_ratio

include("data.jl")
include("bases.jl")
include("task.jl")

end#module
