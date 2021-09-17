### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 33c40310-f05d-11eb-066c-87511fe4fa53
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics
	gr()
	LinearAlgebra.BLAS.set_num_threads(8)
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ 194bd5e8-3953-44d2-bec0-72e8cba07e81
md"""
### Noisy chain

Consider two atoms with lattice spacing $s$, blockade radius $B$ and position range $\pm \sigma$. So we can say:

$$\begin{align}
	X &\sim \mathcal{U}[-\sigma, \sigma]\\
	Y &\sim \mathcal{U}[s-\sigma, s+\sigma]\\
\end{align}$$

Thus we can calculate the probability of these two atoms interfering to be:

$$\begin{align}
	P(|X-Y|\leq B) = \frac{1}{2σ} \left( \min(B,2\sigma + s) + \max(-B, \min(B,2\sigma-s) \right)
\end{align}$$
"""

# ╔═╡ 4726c0cf-0b39-4bd2-b373-8cd539a3946f
begin
	B=1
	P(s,σ) = (min(B,2σ+s) + max(-B, min(B,2σ-s)))/(4σ)
end

# ╔═╡ 56af5dae-c489-4c7f-9131-1c8223623a6a
let s_range = 0.5:(3B/100):7B,
	σ_range = 0:(3B/50):4B
	heatmap(s_range, σ_range, P.(s_range', σ_range); xlabel="s", ylabel="σ")
end

# ╔═╡ b3105074-35c9-4ddf-b94e-9f92abb506ac
plot(0:(3B/50):4B, 1 .- P.(0:(3B/50):4B, 0:(3B/50):4B))

# ╔═╡ 17b0fd1a-6213-457d-ad42-ae3176e313c1
begin
	function successrate(geom, N; trys=10000)
		count = trys
		for _ in 1:trys
			try
				sample_blockaded(geom, N)
			catch e;
				count -= 1
			end
		end
		count / trys
	end
	function chain_successrate(N, s, σ; trys=1000)
		successrate(NoisyChainPBC(N, s, σ), N)
	end
end

# ╔═╡ c4c39853-6609-4309-9408-07a0cc64fe7c
plot(0:(3B/50):4B, chain_successrate.(Ref(2), 0:(3B/50):4B, 0:(3B/50):4B))

# ╔═╡ 709214e0-de51-482d-96d7-84fec5a0f1d7
chain_successrate(10,1.2,0.5)

# ╔═╡ Cell order:
# ╠═33c40310-f05d-11eb-066c-87511fe4fa53
# ╠═194bd5e8-3953-44d2-bec0-72e8cba07e81
# ╠═4726c0cf-0b39-4bd2-b373-8cd539a3946f
# ╠═56af5dae-c489-4c7f-9131-1c8223623a6a
# ╠═b3105074-35c9-4ddf-b94e-9f92abb506ac
# ╠═c4c39853-6609-4309-9408-07a0cc64fe7c
# ╠═17b0fd1a-6213-457d-ad42-ae3176e313c1
# ╠═709214e0-de51-482d-96d7-84fec5a0f1d7
