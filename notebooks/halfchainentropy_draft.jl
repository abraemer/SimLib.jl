### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ f13a03a0-24e6-11ec-2825-8123926abe5d
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics, SpinSymmetry, LsqFit, SparseArrays
	gr()
	LinearAlgebra.BLAS.set_num_threads(8)
	#cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ 3eef0c5c-9ef5-4e0e-ab9b-0721e68f8413
using BenchmarkTools

# ╔═╡ 25cbd399-a199-4469-be1d-f402df522511
md"""
# Entanglement Entropy implementation
## Basics
Have density matrix $\rho = \sum_i p_i |i\rangle\langle i|$ on full system. Can divide system in subsystems $A$ and $B$.

### Partial trace
```math
\rho_B = \mathrm{Tr}_A \rho = \sum_{j} \sum_i \langle a_i |\langle b_j| \rho |a_i \rangle b_j \rangle |b_j\rangle\langle b_j| = \sum_{i,j} \langle a_i b_j| \rho |a_i b_j \rangle |b_j\rangle\langle b_j|
```
where $\{a_i\}$ ($\{b_i\}$) are a basis of $A$ ($B$).

### Von-Neumann entropy:
```math
S(\rho) = \mathrm{Tr} \rho \log_2 \rho = \sum_i p_i \log_2 p_i
```

### Entanglement entropy
Entanglement entropy between subsystem $A$ and $B$ is
```math
S_{A} = S(\rho_A) = S(\mathrm{Tr}_B \rho) = S(\rho_B) = S_B
```
→ Need to compute partial trace and compute eigenvalues.

→ Better way for pure states: SVD
```math
|\psi\rangle = \sum_{i,j} c_{i,j} |a_i,b_j\rangle \overset{SVD}{=} \sum_i \sigma_i |\tilde{a}_i\rangle|\tilde{b}_i\rangle
```
Here we basically *reshaped* the state vector into a matrix and used singular value decomposition on it. This allows easy computation of the eigenvalues of the reduced density matrix. Consider:
```math
\begin{align}
\rho_B = \mathrm{Tr}_A \rho &= \mathrm{Tr}_A \sum_{i,j} \sigma_i \sigma_j^\star |\tilde{a}_i\rangle|\tilde{b}_i\rangle \langle\tilde{a}_j |\langle\tilde{b}_j|\\
&= \sum_j |\sigma_j|^2 |\tilde{b}_j\rangle \langle \tilde{b}_j|
\end{align}
```
**Note:** This also shows that $S(\rho_A) = S(\rho_B)$.

**Note 2:** The SVD way is computationally much faster than partial trace + eigen decomposition and saves a lot of memory as well.
"""

# ╔═╡ 3f654c7b-ff3b-423c-91ad-8c942ca5ece9
md"""
## Additional symmetries
For this computation having $|\psi\rangle$ in a subspace of a symmetry does not make the computation more straight-forward as the *reshape* operation does not work too well in all symmetrized bases.

What follows is explicitly only for spin-1/2 chains of $N$ spins but can probably be generalized more or less easily.

### z-Block basis
Let's denote the eigenspaces of the magnetization $\hat{S}_z$ by $L_k^N$ (where $k$ counts the number of $\uparrow$ in that space) and states by $|k, i\rangle$. The second index just enumerates the different basis vectors in this sector. The whole Hilbert space $\mathcal{H}$ decomposes:
```math
\mathcal{H} = \bigoplus_{k=0}^{N} L^N_k
```

Assume $|\psi\rangle \in L_k$ and thus $|\psi\rangle = \sum_i c_i |k,i\rangle$ and we want to trace out the first $N_1$ spins of the chain.

In this case we are lucky because $L^N_k$ again decompose nicely into smaller magnetization eigenspaces:
```math
L_k^N = \bigoplus_{k_1=k_{min}}^{k_{max}} L^{N_1}_{k_1} \otimes L^{N-N_1}_{k-k_1}
```
with $k_{min} = \max\{0, N_1+k-N\}$ and $k_{max} = \min\{k, N_1\}$.

This means, that the reduced density matrix is actually block diagonal! And the computation of the Schmidt values can be simplified as well as for every $k$ we again can reshape the corresponding vectors into a rectangular matrix.

```math
|\psi\rangle = \sum_i c_i |k,i\rangle = \sum_{k_1=k_{min}}^{k_{max}} \sum_{i,j} c_{i,j} |k_1,i\rangle|k-k_1,j\rangle
```

**Implementation note:**
The only thing left is to determine the correct index $i \in L^N_k$ from $(i,j) \in L^{N_1}_{k_1} \otimes L^{N-N_1}_{k-k_1}$. I currently do this via a lookup table and some index math. I already have functions to compute the indices of $L^N_k \subset \mathcal{H}$. 

So what I do is: 
1. Compute the indices $I$ and $J$ of $L^{N_1}_{k_1} \subset L^N_k$ and $L^{N-N_1}_{k-k_1} \subset L^N_k$
2. This gives the indices of $L^{N_1}_{k_1} \otimes L^{N-N_1}_{k-k_1} \subset L^N_k$ by $I\otimes J = \{2^{N-N_1}i+j | i\in I, j\in J\}$. 
3. Thus we have the subspace's indices $I\otimes J \subset L^N_k \subset \mathcal{H}$
4. This I map via lookup table to the actual indices of the state vector.
"""

# ╔═╡ fc4f5bbb-cc20-45ed-aa1a-928ae12b1566
md"""
### Parity (spin flip)

This is not as easy although we can find a decomposition. Let $P^N_\pm$ denote the symmetry sectors. We can see:
```math
P^N_+ = P^{N_1}_+ \otimes P^{N-N_1}_+ \oplus P^{N_1}_- \otimes P^{N-N_1}_-
```

Again we have 2 blocks not talking to each other that may be treated seperately. Finding the indices is a bit more involved this time around.

Consider a state $|s\rangle \in \mathcal{H} = |s_1, s_2\rangle$ and its flipped partner $|\bar{s}\rangle = |\bar{s}_1, \bar{s}_2\rangle$. Also consider the half-flipped state $|s^\star\rangle = |s_1, \bar{s}_2\rangle$ and its partner $|\bar{s}^\star\rangle = |\bar{s}_1, s_2\rangle$. Thus we have $|S\rangle = |s\rangle + |\bar{s}\rangle \in P^N_+$ and $|S^\star\rangle = |s^\star\rangle + |\bar{s}^\star\rangle \in P^N_+$. One can directly construct:
```math
\begin{align}
|t_+\rangle &= |S\rangle + |S^\star\rangle = (|s_1\rangle + |\bar{s}_1\rangle)\otimes (|s_2\rangle + |\bar{s}_2\rangle) \in P^{N_1}_+ \otimes P^{N-N_1}_+\\
|t_-\rangle &= |S\rangle - |S^\star\rangle = (|s_1\rangle - |\bar{s}_1\rangle)\otimes (|s_2\rangle - |\bar{s}_2\rangle) \in P^{N_1}_- \otimes P^{N-N_1}_-
\end{align}
```
"""

# ╔═╡ 119fb04d-219b-4a08-a50b-584e470f517b
N = 11

# ╔═╡ 6ad0d260-5284-41f2-b7d7-273023d9114c
H = xxzmodel(Symmetric((1 .+ (2*rand(N,N))) .^ 6),-0.7)

# ╔═╡ 942f4e4c-8f21-4a60-ae78-e183e003c748
eig = eigen!(Hermitian(Matrix(H)));

# ╔═╡ a29ef3cf-b4c1-4320-b580-f3cda39e7fac
eig.vectors[:, 5]

# ╔═╡ dd2a8d83-7251-402c-81ea-7b9e52138080
size(eig.vectors,1)

# ╔═╡ 0ea97d53-929e-4683-9742-3183a527abf4
k = 5

# ╔═╡ 4539ba8a-ec44-444d-8ad0-ccc07e7443f5
abs2.(svdvals(reshape(eig.vectors[:, k], 2^div(N-1,2), 2^(N-div(N-1,2)))))

# ╔═╡ f7145441-e14b-47dc-b8c4-c7d243b38c8e
entropy(abs2.(svdvals(reshape(eig.vectors[:, k], 2^div(N-1,2), 2^(N-div(N-1,2))))))

# ╔═╡ 6472d25b-94f7-42ee-8a9d-935b31748eb9
eig.values[:,1]

# ╔═╡ 9bf89fcc-81f9-40c2-9fa5-ddd2eea47941
function halfchain_entropy(state, N=nspins(state), N1=div(N,2))
	entropy(abs2.(svdvals(reshape(state, 2^N1, 2^(N-N1)))))
end

# ╔═╡ d9b06c60-56be-4128-9422-fbfcda7f53e8
plot(halfchain_entropy.(eachcol(eig.vectors)))

# ╔═╡ 2bf00bc0-1d51-4134-afe3-9d5cf897f495
halfchain_entropy.(eachcol(eig.vectors))

# ╔═╡ 0f9e0ab5-7c6e-4e7a-9e58-2185c0b55e4f
halfchain_entropy(eig.vectors[:, k])

# ╔═╡ 1274618f-d59e-4af7-ad6a-81e9c4b1e8bd
begin
	function partial_trace(vec)
		N = round(Int, log2(length(vec)))
		Nhalf = div(N,2)
		ρ = vec * vec'
		tmp = reshape(ρ, 2^Nhalf,2^(N-Nhalf), 2^Nhalf,2^(N-Nhalf))
		res = zeros(eltype(ρ), 2^Nhalf, 2^Nhalf)
		for i in 1:2^(N-Nhalf)
			res += tmp[:,i,:,i]
		end
		Hermitian(res)
	end
end

# ╔═╡ 6910adba-dea2-4095-b888-12bf5a259fd5
reverse(eigvals(partial_trace(eig.vectors[:,k])))

# ╔═╡ efc5f35f-bb74-40ab-b335-b91a1ac7b6b6
NB = N

# ╔═╡ bf58d3eb-46fe-41b8-819a-1827bb29ab51
kB = div(N-1,2)

# ╔═╡ 18820e48-e53b-4d06-ac56-a2a958737156
N1 = 3

# ╔═╡ ae78b8ca-2a42-401d-ac1e-1a0507d015a1
symm_basis = ZBlockBasis(N,kB)

# ╔═╡ 1a6fdac3-7f6f-4c7b-b379-977268b54707
ψsymm = normalize!(ones(basissize(symm_basis)));

# ╔═╡ ae734d90-65cf-4e37-9c6c-aa725d31131c
ψfull = Vector(sparsevec(SpinSymmetry._indices(symm_basis), ψsymm, 2^NB));

# ╔═╡ 44a313c0-3c1b-4769-9176-81acbeffb017
entropy(abs2.(svdvals(reshape(ψfull, 2^N1, 2^(NB-N1)))))

# ╔═╡ 1957e200-71fe-45e3-aee5-39b8eecba31a
SZ = sum(op_list(σz, NB))

# ╔═╡ fff4c13f-003c-49f8-968a-a11a984ab8b0
diag(SZ[SpinSymmetry._indices(symm_basis), SpinSymmetry._indices(symm_basis)])

# ╔═╡ 70d4e7b4-4149-4094-ae75-9c6e7eb097ea
SpinSymmetry._indices(symm_basis)

# ╔═╡ 90000a7c-e927-4de3-b7ab-f238bd5f7af5
indexLookup = let inds = SpinSymmetry._indices(symm_basis)
	sparsevec(inds, 1:length(inds))
end

# ╔═╡ c1aac45a-ca0a-4188-af09-b082ae14cd7c
let res = []
	for k in reverse(0:min(N1,kB))
		let b1 = SpinSymmetry._indices(ZBlockBasis(N1,k)),
			b2 = SpinSymmetry._indices(ZBlockBasis(NB-N1, kB-k))
			push!(res, vec(b1 .+  2^N1 * (b2 .- 1)'))
		end
	end
	vcat(res...)
end

# ╔═╡ b121dd76-ecde-415b-8d29-e92c452b305b
let inds = SpinSymmetry._indices(symm_basis),
	v = Dict((i, ind) for (ind, i) in enumerate(inds))
	@btime $v[118]
end

# ╔═╡ db4667e3-0ed5-430b-b9e2-b245608d8d7b
v = Dict((i, ind) for (ind, i) in enumerate(SpinSymmetry._indices(symm_basis)))

# ╔═╡ 5db25eaa-8fe0-45ed-9029-a42ebc968c63
let v = Vector(indexLookup), 
	k = [56,111], res = zeros(Int, length(k))
	@btime $res .= getindex.(Ref($v), $k)
end

# ╔═╡ b9928ff9-8c98-4498-b1d3-3597357a1ffd
let v = Dict((i, ind) for (ind, i) in enumerate(SpinSymmetry._indices(symm_basis))),
	k = [56,111], res = zeros(Int, length(k))
	@btime $res .= getindex.(Ref($v), $k)
end

# ╔═╡ 1e976790-2e7c-4e43-a222-a1e0d13c4f7f
let v = indexLookup,
	k = [56,111], res = zeros(Int, length(k))
	@btime $res .= getindex.(Ref($v), $k)
end

# ╔═╡ ea33e44e-65fd-46a7-b077-cfac2a0ef768
allinds = 
	[SpinSymmetry._indices(ZBlockBasis(N1,k)) .+ 
		2^N1 .* (SpinSymmetry._indices(ZBlockBasis(NB-N1, kB-k)) .- 1)'
		 for k in max(0,N1+kB-NB):min(N1,kB)]


# ╔═╡ 3c1b94c5-2279-40bf-8b69-3b4dac8ea490
reducedρ2 = let res = [], indexLookup = Vector(indexLookup)
	for indmat in allinds
		l = size(indmat, 1)
		temp = zeros(eltype(ψsymm), l, l)
		vals = zeros(eltype(ψsymm), l)
		for inds in eachcol(indmat)
			vals .= getindex.(Ref(ψsymm), getindex.(Ref(indexLookup), inds))
			temp = mul!(temp, vals, vals', 1, 1)
		end
		push!(res, temp)
	end
	res
end

# ╔═╡ 5c262eea-0622-4046-a60d-432f24daf651
eigvals.(reducedρ2)

# ╔═╡ 7cc24ef6-a45b-4f87-a17f-91f7b196f5ad
svdvals(ψsymm[getindex.(Ref(indexLookup), allinds[2])]).^2

# ╔═╡ a84a4a13-ba18-4697-a643-817147e597bd
typeof(allinds)

# ╔═╡ 49481125-553b-48df-907b-ef98051cac47
sum(length, allinds)

# ╔═╡ 7f0db04e-5701-4978-bc8c-e98753bbb749
reducedρ = let res = []
	for k in reverse(0:min(N1,kB))
		let b1 = SpinSymmetry._indices(ZBlockBasis(N1,k)),
			b2 = 2^N1 * (SpinSymmetry._indices(ZBlockBasis(NB-N1, kB-k)) .- 1),
			temp = zeros(eltype(ψsymm), length(b2), length(b2))
			for indA in b1
				fullBasisInds = indA .+ b2
				symmBasisInds = indexLookup[fullBasisInds]
				vals = ψsymm[symmBasisInds]
				temp += vals * vals'
			end
			push!(res, temp)
		end
	end
	res
end

# ╔═╡ 2ba4c3c4-ec8a-4791-b940-4251fa9ae145
entropy(vcat(svdvals.(reducedρ)...))

# ╔═╡ 8cfdd892-56c1-49f8-a993-2f426f9ec690
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

# ╔═╡ 4b3461ca-73e6-4c1a-819a-53ab3774a6f4
entanglement_entropy_zblock(ψsymm, N, kB, N1)

# ╔═╡ c1343cbc-37b9-446e-9035-57983934c99b
Hsymm = symmetrize_operator(H, ZBlockBasis(N, kB))

# ╔═╡ 2cf277f4-38f9-43c6-90b4-20c1a0c5a5df
eigsymm = eigen!(Hermitian(Matrix(Hsymm)));

# ╔═╡ 98f5eff9-d71b-4a2d-ab88-a88b9fffa944
plot(entanglement_entropy_zblock.(eachcol(eigsymm.vectors), N, div(N-1,2), kB))

# ╔═╡ 79893553-b8b2-4d90-80f8-3b0d01d5096d


# ╔═╡ Cell order:
# ╠═25cbd399-a199-4469-be1d-f402df522511
# ╠═3f654c7b-ff3b-423c-91ad-8c942ca5ece9
# ╠═fc4f5bbb-cc20-45ed-aa1a-928ae12b1566
# ╠═f13a03a0-24e6-11ec-2825-8123926abe5d
# ╠═119fb04d-219b-4a08-a50b-584e470f517b
# ╠═6ad0d260-5284-41f2-b7d7-273023d9114c
# ╠═942f4e4c-8f21-4a60-ae78-e183e003c748
# ╠═a29ef3cf-b4c1-4320-b580-f3cda39e7fac
# ╠═dd2a8d83-7251-402c-81ea-7b9e52138080
# ╠═0ea97d53-929e-4683-9742-3183a527abf4
# ╠═4539ba8a-ec44-444d-8ad0-ccc07e7443f5
# ╠═f7145441-e14b-47dc-b8c4-c7d243b38c8e
# ╠═d9b06c60-56be-4128-9422-fbfcda7f53e8
# ╠═6472d25b-94f7-42ee-8a9d-935b31748eb9
# ╠═2bf00bc0-1d51-4134-afe3-9d5cf897f495
# ╠═0f9e0ab5-7c6e-4e7a-9e58-2185c0b55e4f
# ╠═9bf89fcc-81f9-40c2-9fa5-ddd2eea47941
# ╠═6910adba-dea2-4095-b888-12bf5a259fd5
# ╠═1274618f-d59e-4af7-ad6a-81e9c4b1e8bd
# ╠═efc5f35f-bb74-40ab-b335-b91a1ac7b6b6
# ╠═bf58d3eb-46fe-41b8-819a-1827bb29ab51
# ╠═18820e48-e53b-4d06-ac56-a2a958737156
# ╠═1a6fdac3-7f6f-4c7b-b379-977268b54707
# ╠═ae734d90-65cf-4e37-9c6c-aa725d31131c
# ╠═44a313c0-3c1b-4769-9176-81acbeffb017
# ╠═2ba4c3c4-ec8a-4791-b940-4251fa9ae145
# ╠═ae78b8ca-2a42-401d-ac1e-1a0507d015a1
# ╠═1957e200-71fe-45e3-aee5-39b8eecba31a
# ╠═fff4c13f-003c-49f8-968a-a11a984ab8b0
# ╠═70d4e7b4-4149-4094-ae75-9c6e7eb097ea
# ╠═90000a7c-e927-4de3-b7ab-f238bd5f7af5
# ╠═c1aac45a-ca0a-4188-af09-b082ae14cd7c
# ╠═3eef0c5c-9ef5-4e0e-ab9b-0721e68f8413
# ╠═b121dd76-ecde-415b-8d29-e92c452b305b
# ╠═db4667e3-0ed5-430b-b9e2-b245608d8d7b
# ╠═5db25eaa-8fe0-45ed-9029-a42ebc968c63
# ╠═b9928ff9-8c98-4498-b1d3-3597357a1ffd
# ╠═1e976790-2e7c-4e43-a222-a1e0d13c4f7f
# ╠═ea33e44e-65fd-46a7-b077-cfac2a0ef768
# ╠═3c1b94c5-2279-40bf-8b69-3b4dac8ea490
# ╠═5c262eea-0622-4046-a60d-432f24daf651
# ╠═7cc24ef6-a45b-4f87-a17f-91f7b196f5ad
# ╠═a84a4a13-ba18-4697-a643-817147e597bd
# ╠═49481125-553b-48df-907b-ef98051cac47
# ╠═7f0db04e-5701-4978-bc8c-e98753bbb749
# ╠═8cfdd892-56c1-49f8-a993-2f426f9ec690
# ╠═4b3461ca-73e6-4c1a-819a-53ab3774a6f4
# ╠═c1343cbc-37b9-446e-9035-57983934c99b
# ╠═2cf277f4-38f9-43c6-90b4-20c1a0c5a5df
# ╠═98f5eff9-d71b-4a2d-ab88-a88b9fffa944
# ╠═79893553-b8b2-4d90-80f8-3b0d01d5096d
