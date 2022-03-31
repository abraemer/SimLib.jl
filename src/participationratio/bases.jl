abstract type PRBasis end

function construct_basis end

struct ZBasis <: PRBasis end

referencebasis_name(::ZBasis) = "zbasis"


abstract type PairingMethod end

function permutation_matrix end

struct PetersMethod <: PairingMethod end

referencebasis_name(::PetersMethod) = "pairs"

# pair up nearest neighbours like
# (1,2) (3,4) (4,5) ...
# for shifted=true (N,1) (2,3) (4,5)...
struct NaivePairing{SHIFTED} <: PairingMethod
    NaivePairing(shifted::Bool) = new{shifted}()
end

referencebasis_name(::NaivePairing{false}) = "naivepairs"
referencebasis_name(::NaivePairing{true}) = "naivepairs_shifted"


peters_clustering(J) = peters_clustering!(copy(J))
function peters_clustering!(J)
    N = size(J,1)
    @assert iseven(N)
    @assert N == size(J,2) # square matrix

    clusters = Vector{Int}[]
    Base.sizehint!(clusters, div(N,2))
    while length(clusters) < div(N,2)
        I = argmax(J)
        a,b = I.I
        push!(clusters, [a,b])
        J[a,:] .= 0
        J[b,:] .= 0
        J[:,a] .= 0
        J[:,b] .= 0
    end
    return clusters
end

# perm: p[k] -> at the kth position goes the element formely at p[k]
#iperm: ip[k] -> the kth element was formely at position ip[k]
"""
    permutation_to_swaps!(p; compress=true)

Convert a permutation `p`, encoded as vector, to a series of swaps, also encoded as vector.
Destroys `p` in the process, so do not use it afterwards!

If `compress=true` then stop once the remaining elements are sorted.

**Example:**
permutation_to_swaps!([2,1,3]; compress=false) = [2,2,3] # swap 1-2 then 2-2 then 3-3
permutation_to_swaps!([2,1,3]; compress=true) = [2] # swap 1-2 and done
"""
function permutation_to_swaps!(p; compress=false)
	ip = invperm(p)
	swaps = Int[]
	for i in 1:length(p)
        compress && issorted(view(p, i:length(p))) && return swaps
		pi = p[i]
		push!(swaps, pi) # swap positions i and p[i]
		# update permutation
		# at position p[i] is now i
		# so need to swap p[i] and p[ip[i]]
		p[i],p[ip[i]] = p[ip[i]], p[i]
		ip[i],ip[pi] = ip[pi], ip[i] # and keep pi in sync
	end
	return swaps
end


function swaps_to_matrix(swaps, Nspins)
	matrix_elements = 2^Nspins
	J = collect(1:matrix_elements)
	I = apply_swaps.(Ref(swaps), J)
	V = ones(matrix_elements)
	return sparse(I,J,V)
end

function apply_swaps(swaps,i)
	i -= 1
	for (j,k) in enumerate(swaps)
		i = SpinSymmetry._swap_bits(i,j-1,k-1)
	end
	return i+1
end

function permutation_matrix(::PetersMethod, J)
    # sort the clustering such that it's nicer to look at
    clustering = sort(sort.(peters_clustering(J)))
    basis_ordering = reduce(vcat, clustering)
    return swaps_to_matrix(permutation_to_swaps!(basis_ordering), size(J,1))
end

permutation_matrix(::NaivePairing{false},J) = I
permutation_matrix(::NaivePairing{true},J) = swaps_to_matrix(2:size(J,1), size(J,1))

struct PairBasis{M<:PairingMethod} <: PRBasis
    Npairs::Int
    pairingmethod::M
    function PairBasis(N, method::PairingMethod)
        @assert iseven(N)
        return new{typeof(method)}(N÷2, method)
    end
end

referencebasis_name(pb::PairBasis) = referencebasis_name(pb.pairingmethod)

function construct_basis(pb::PairBasis)
    matrix_elements = 6^pb.Npairs
    I,J,V = zeros(Int64, matrix_elements), zeros(Int64, matrix_elements), zeros(matrix_elements)
    I[1:6] .= [0,1,1,2,2,3]
	J[1:6] .= [0,1,2,1,2,3]
	V[1:6] .= [1,√(0.5),√(0.5),√(0.5),-√(0.5),1]
    for k in 1:pb.Npairs-1
        len = 6^k # current subblock length
        block = [k*len+1:(k+1)len for k in 0:5]
        spin1UP = 2^(2k)
        spin2UP = 2^(2k+1)
        # ↓↓ is fine
        I[block[2]] .= spin1UP .+ I[block[1]]
        J[block[2]] .= spin1UP .+ J[block[1]]
        V[block[2]] .= √(0.5) .* V[block[1]]

        I[block[3]] .= spin1UP .+ I[block[1]]
        J[block[3]] .= spin2UP .+ J[block[1]]
        V[block[3]] .= √(0.5) .* V[block[1]]

        I[block[4]] .= spin2UP .+ I[block[1]]
        J[block[4]] .= spin1UP .+ J[block[1]]
        V[block[4]] .= √(0.5) .* V[block[1]]

        I[block[5]] .= spin2UP .+ I[block[1]]
        J[block[5]] .= spin2UP .+ J[block[1]]
        V[block[5]] .= -√(0.5) .* V[block[1]]

        I[block[6]] .= spin1UP + spin2UP .+ I[block[1]]
        J[block[6]] .= spin1UP + spin2UP .+ J[block[1]]
        V[block[6]] .= V[block[1]]
    end
    # shift indices from 0:2^N-1  to 1:2^N
    I .+= 1
    J .+= 1
    return sparse(I,J,V)
end
