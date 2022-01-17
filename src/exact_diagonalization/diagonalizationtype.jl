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

struct ShiftInvertARPACK <: SparseMethod end
struct ShiftInvert <: SparseMethod end
struct POLFED <: SparseMethod end

struct Sparse{M<:SparseMethod} <: DiagonalizationType
    method::M
    σ::Vector{Float64}
    count::Int64
    normalize::Bool
end
#Base.:(==)(s1::Sparse, s2::Sparse) = s1.σ == s2.σ && s1.count == s2.count

spectral_size(sp::Sparse, model) = sp.count
ed_size(sp::Sparse) = length(sp.σ)

function diagonalize!(callback, H, sp::Sparse{ShiftInvertARPACK})
    if sp.normalize
        emin, emax = eigsolve(H, 1, :SR)[1][1], eigsolve(H, 1, :LR)[1][1]
        H .= (H - emin*I) ./ (emax-emin)
    end
    for (i, σ) in enumerate(sp.σ)
        (d,v,nconv,niter,nmult,resid) = eigs(H; nev=sp.count, sigma=σ, check=1, tol=1e-12)
        logmsg("$nconv converged after $niter iterations.")
        callback(i, LinearAlgebra.sorteig!(d,v)...)
    end
end

function diagonalize!(callback, H, sp::Sparse{ShiftInvert})
    if sp.normalize
        emin, emax = eigsolve(H, 1, :SR)[1][1], eigsolve(H, 1, :LR)[1][1]
        H .= (H - emin*I) ./ (emax-emin)
    end

    L = size(H, 1)
    for (i, σ) in enumerate(sp.σ)
        iH = let F = factorize(H-σ*I)
            #LinearMap{eltype(H)}(x -> F \ x, L, ismutating=false, issymmetric=true)
            x -> F \ x
        end
        evals, evecs, info = eigsolve(iH, L, sp.count; ishermitian=true, krylovdim=max(20,2*sp.count+1))
        @. evals = σ + 1 / evals
        logmsg("Shift-invert: $(info.converged) converged after $(info.numiter) iterations.")
        callback(i, LinearAlgebra.sorteig!(evals, hcat(evecs...))...)
    end
end


function diagonalize!(callback, H, sp::Sparse{POLFED})
    emin, emax = eigsolve(H, 1, :SR)[1][1], eigsolve(H, 1, :LR)[1][1]

    if sp.normalize
        H .= (H - emin*I) ./ (emax-emin)
        emin, emax = 0.0, 1.0
    end

    L = size(H, 1)
    Δϵ = (emax-emin)/size(H, 1) # average level spacing

    for (i, σ) in enumerate(sp.σ)
        # this order estimate needs to be improved!
        order = order_estimate(σ, Δϵ, (emin,emax))
        δH = delta(H, σ, (emin, emax); order)
        _, vecs, info = eigsolve(δH, L, sp.count; ishermitian=true, krylovdim=max(20,2*sp.count+1))
        logmsg("POLFED with order $order: $(info.converged) converged after $(info.numiter) iterations.")
        # compute eigenvals from vectors:
        vals = dot.(vecs, Ref(H), vecs) ./ dot.(vecs, vecs)

        callback(i, LinearAlgebra.sorteig!(vals, hcat(vecs...)))
    end
end
