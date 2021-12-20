abstract type DiagonalizationType end


function ed_size end
function spectral_size end
function diagonalize! end

struct Full <: DiagonalizationType end
Base.:(==)(::Full, ::Full) = true

ed_size(::Full) = 1
spectral_size(::Full, model) = basissize(model.basis)

function diagonalize!(callback, H, ::Full)
    callback(1, eigen!(Hermitian(Matrix(H)))...)
end



abstract type SparseMethod end

function sparse_diagonalize end


struct Sparse{M<:SparseMethod} <: DiagonalizationType
    method::M
    σ::Vector{Float64}
    count::Int64
    normalize::Bool
end
#Base.:(==)(s1::Sparse, s2::Sparse) = s1.σ == s2.σ && s1.count == s2.count

spectral_size(sp::Sparse, model) = sp.count
ed_size(sp::Sparse) = length(sp.σ)

function diagonalize!(callback, H, sp::Sparse)
    if sp.normalize
        #TODO
        emin, emax = 1,0
        H .= (H - emin*I) ./ (emax-emin)
    end
    for (i, σ) in sp.σ
        eigen = sparse_diagonalize(sp.method, H, σ, sp.count)
        callback(i, eigen...)
    end
end

struct ShiftInvert <: SparseMethod

end

struct POLFED <: SparseMethod end

function sparse_diagonalize(::POLFED, H, count)
    ## TODO
end
