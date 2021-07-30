### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5e92c53e-e948-11eb-17ce-5b1baf22cbf9
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics
	gr()
	LinearAlgebra.BLAS.set_num_threads(8)
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ c63ad93a-bbe0-4771-a2c7-fca17ac5cb44
data_zero_field = SimLib.ED.load("/home/adrian/results/julia-cusp/field-section/data/ed_noisy_chain_pbc_1d_alpha_6.0_N_13-k_6.jld2").evals[:,:,:,[11,21,end]];

# ╔═╡ bdab9d57-4c6e-4e55-8037-e29ec55c792c
#data = SimLib.ED.load(path_prefix(), :noisy_chain_pbc, 9, 1, 6);
data = SimLib.ED.load("/home/adrian/results/julia-cusp/", :noisy_chain_pbc, 13, 1, 6);

# ╔═╡ 5493eab2-85ec-4805-9f2f-a6a447f9d16b
md"""
| Parameter | Slider |
| :-------- | :----: |
| Density   | $(@bind rhoIndex PlutoUI.Slider(1:length(SimLib.ED.ρ_values(data))))|
| Shot      | $(@bind shotIndex PlutoUI.Slider(1:SimLib.ED.shots(data)))|
| Field     | $(@bind fieldIndex PlutoUI.Slider(1:length(SimLib.ED.fields(data))))|
"""

# ╔═╡ d4fb5f55-37ac-49ab-b055-cb8a326cb3c9
md"""##### ρ= $(SimLib.ED.ρ_values(data)[rhoIndex]), h= $(SimLib.ED.fields(data)[fieldIndex]), shot= $shotIndex"""

# ╔═╡ 2b14057b-997b-45ad-a855-660165a55a1d
let
	a=5
	## energy
	plot(data.evals[:,shotIndex,fieldIndex,rhoIndex]; 
		label="Energy")
	## mag
	plot!([0];color="purple", label="magnetization")
	plot!(twinx(), mean(data.eev[:,:,shotIndex,fieldIndex,rhoIndex]; dims=1)[1,:];
		legend=nothing, axis=:right, color="purple", ylabel="magnetization")
	## EON
	plot(data.eon[:,shotIndex,fieldIndex,rhoIndex]; 
		label="eigenstate occupation")
	## E_0
	#hline!([dot(data.evals[:,shotIndex,fieldIndex,rhoIndex],
	#	data.eon[:,shotIndex,fieldIndex, rhoIndex])]; label=L"$\langle E \rangle$")
end

# ╔═╡ 749e5dde-3b8f-4b3d-95b1-427d270412df
begin
	scatter(data.evals[:,shotIndex,fieldIndex,rhoIndex],
		mean(data.eev[:,:,shotIndex,fieldIndex,rhoIndex]; dims=1)[1,:]; 
		title="Mag vs. Energy", xlabel="Energy", ylabel="Magnetization")
end

# ╔═╡ 2fa6a9c7-c0e7-4ac6-ba68-a6522349fb77
let N = SimLib.ED.system_size(data),
	nHilbert = size(data.evals,1),
	ratios = levelspacingratio(data.evals; center_region=0.75),
	ratios_zero_field = levelspacingratio(data_zero_field; center_region=0.75),
	r_mean = dropdims(mean(ratios; dims=(1,2)); dims=(1,2)),
	r_std  = dropdims(std(ratios; dims=(1,2)); dims=(1,2)) / (size(ratios,2))^0.5,
	r_zero_field = dropdims(mean(ratios_zero_field; dims=(1,2)); dims=(1,2)),
	r_zero_field_std  = dropdims(std(ratios_zero_field; dims=(1,2)); dims=(1,2)) / (size(ratios,2))^0.5
	
	# take <r> per shot and compute stddev on those
	
	p = plot(;title="Mean level spacing ratios", legend=:left)
	for (i, ρ) in enumerate(data.ρs)
		plot!(data.fields, r_mean[:,i]; label="ρ=$ρ")#, ribbon=r_std[:,i])
		hline!(r_zero_field[:,i]; label="ρ=$ρ", ls=:dot, 
			ribbon=r_zero_field_std[:,i]/10,
			color=p.series_list[end].plotattributes[:seriescolor])
	end
	hline!([0.5295]; label="GOE", ls=:dash, width=2)
	hline!([2 * log(2)-1]; label="Poisson", ls=:dash, width=2)
	#plot!(data.fields, rZBlock; label="single z-block", ls=:dot, width=2)
	p
end

# ╔═╡ a5293de9-b76a-4a38-80ed-c4794e56959e
begin
	wigner(x; β=1, Z_β=8/27) = 1/Z_β * (x+x^2)^β / (1+x+x^2)^(1+1.5β)
	poisson(r) = 1/(1+r)^2
end

# ╔═╡ 4ca67e8c-f712-43a4-ae6a-27d51c6c88ef
let	N = size(data.evals, 1),
	middle = Int(N/2-N/4):Int(N/2+N/4)-1,
	cutoff = 1,
	npoints = 100,
	xspace = 0:(cutoff/(npoints-1)):cutoff,
	levels = data.evals[middle,:, fieldIndex, rhoIndex],
	spacings = levels[2:end,:] .- levels[1:end-1,:],
	ratios_raw = sort!((spacings[2:end,:] ./ spacings[1:end-1,:])[:]),
	mask = (ratios_raw .< cutoff),
	ratios = min.(ratios_raw, 1 ./ ratios_raw),
	wigner_prediction = wigner.(xspace),
	poisson_prediction = poisson.(xspace),
	scale = sum(mask)/length(mask)
	
	histogram(ratios[mask]; bins=npoints, normalize=true)
	plot!(xspace, poisson_prediction/scale; width=2, label="Poisson")
	plot!(xspace,  wigner_prediction/scale; width=2, label="Wigner")
end

# ╔═╡ 27a1bca1-bbfc-41ef-a6f1-6b1e97ccad4b
function chainJ(N, α)
	J = collect(1.0:N) .- collect(1:N)'
	J[diagind(J)] .= 1
	J .^= -α
	J[diagind(J)] .= 0
	J
end

# ╔═╡ c5f003a8-b94a-4321-9c94-476cefa2702c
function levelspacingratio_mean(levels; center_only=false)
	sizes = size(levels)
	L = sizes[1]
	range = center_only ? (div(L,4)+2:3*div(L,4)) : 3:L
	@show range
	res = Array{Float64, length(sizes)}(undef, 1, sizes[2:end]...)
	for I in CartesianIndices(axes(levels)[2:end])
		for j in range
			ratio = (levels[j-1,I]-levels[j,I])/(levels[j-2,I]-levels[j-1,I])
			res[I] += min(ratio, 1/ratio)
		end
	end
	res ./= length(range)
end

# ╔═╡ e1ce4ac8-2784-4245-9fdb-4aed964f94d4
function levelspacinghistogram(levels; center_only=false, bins=100)
	sizes = size(levels)
	res = zeros(Int, bins, sizes[2:end]...)
	for I in CartesianIndices(axes(levels)[2:end])
		for j in 3:sizes[1]
			ratio_raw = (levels[j-1,I]-levels[j,I])/(levels[j-2,I]-levels[j-1,I])
			ratio = min(ratio_raw, 1/ratio_raw)
			bin = trunc(Int, ratio*bins)+1 # since 0<=ratio <= 1, just multiply
			res[bin, I] += 1
		end
	end
	res
end

# ╔═╡ 283aed70-001b-4958-a169-3d31555d793b
#rZBlock = let N = 13,
#	block_inds = symm_sz_block(6,N),
#	H = symmetrize_op(xxzmodel(chainJ(N, 6), -0.73)),
#	Hblock = H[block_inds, block_inds],
#	field_op = symmetrize_op(fieldterm(N, σx))[block_inds, block_inds]
#	mean.(levelspacingratio(eigvals!(Hermitian(Matrix(Hblock+h*field_op)))) for h in data.fields)
#end

# ╔═╡ 3b393f07-380e-45a7-bd09-4140bd40c42b


# ╔═╡ 7742a2c5-89ba-4498-893a-1c406fc4233e
function shiftIndex(i, N; by=1)
	((i << by) & (2^N-1)) + (i >> (N-by))
end

# ╔═╡ ecaae0ca-8750-4136-9b76-f7872534591e
shiftIndex.(collect(0:2^3-1), 3; by=3)

# ╔═╡ Cell order:
# ╠═5e92c53e-e948-11eb-17ce-5b1baf22cbf9
# ╠═c63ad93a-bbe0-4771-a2c7-fca17ac5cb44
# ╠═bdab9d57-4c6e-4e55-8037-e29ec55c792c
# ╟─5493eab2-85ec-4805-9f2f-a6a447f9d16b
# ╟─d4fb5f55-37ac-49ab-b055-cb8a326cb3c9
# ╟─2b14057b-997b-45ad-a855-660165a55a1d
# ╟─749e5dde-3b8f-4b3d-95b1-427d270412df
# ╟─4ca67e8c-f712-43a4-ae6a-27d51c6c88ef
# ╠═2fa6a9c7-c0e7-4ac6-ba68-a6522349fb77
# ╟─a5293de9-b76a-4a38-80ed-c4794e56959e
# ╟─27a1bca1-bbfc-41ef-a6f1-6b1e97ccad4b
# ╠═c5f003a8-b94a-4321-9c94-476cefa2702c
# ╠═e1ce4ac8-2784-4245-9fdb-4aed964f94d4
# ╠═283aed70-001b-4958-a169-3d31555d793b
# ╠═3b393f07-380e-45a7-bd09-4140bd40c42b
# ╠═ecaae0ca-8750-4136-9b76-f7872534591e
# ╠═7742a2c5-89ba-4498-893a-1c406fc4233e
