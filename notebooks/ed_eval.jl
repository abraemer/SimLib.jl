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
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ bdab9d57-4c6e-4e55-8037-e29ec55c792c
data = SimLib.ED.load(SimLib.path_prefix(), :noisy_chain_pbc, 9, 1, 6);

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

# ╔═╡ b1ee0e64-ce63-443d-b5ec-7448b9c20174
size(data.eon)

# ╔═╡ f19e17d2-26da-4d15-bd23-d666ce0cb066
sum(data.eon) ≈ reduce(*, size(data.eon))/size(data.eon,1)

# ╔═╡ 68b9acb2-5cf9-4576-924e-2699397574ee
ψ0 = vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:10))))

# ╔═╡ 66f9f163-c074-4e9e-941b-c35eec7a0dfb
model = real.(symmetrize_op(xxzmodel(10,1.0, -0.73)));

# ╔═╡ 8d8d6f5d-6360-48f7-a298-fc75e8bed7db
spin_ops = real.(symmetrize_op.(op_list(σx/2, 10)))

# ╔═╡ 634fd7ef-dee7-4917-b237-ca59c4e537f4
field_operator = sum(spin_ops)

# ╔═╡ 94418bc7-1a88-4eb5-a0bd-b11b55fdbfc6
E = eigen!(Hermitian(Matrix(model-1*field_operator)));

# ╔═╡ f74b4aeb-4899-4f9b-862a-b38d8dda6f31
sum(abs2.((E.vectors') * ψ0))

# ╔═╡ dfe5782a-fce2-409d-b7eb-1aabfdcaa18b
size(xxzmodel(10,1,-0.73))d

# ╔═╡ 20ab3428-48bb-4a93-90ad-c961cfea3b64
let N=10,
	n=2^(N-1),
	forward = 1:n,
	bacward = 2^N:-1:n+1,
	H = xxzmodel(N,1,-0.73)
	@show(@views H[forward,forward] == H[bacward,bacward])
	@show(@views H[forward,bacward] == H[bacward,forward])
end

# ╔═╡ c14bc300-4fec-4e49-89b1-d851c37da744
res = abs2.(mul!(zeros(eltype(ψ0), size(ψ0)), E.vectors', ψ0))

# ╔═╡ a491cd5b-e610-4b8c-9a57-8e47494c9c6d
res[argmax(res)], argmax(res)

# ╔═╡ a208de1a-c167-42ed-af6c-1828a61f7d53
function chainJ(N)
	J = collect(1.0:N) .- collect(1:N)'
	J[diagind(J)] .= 1
	J .^= -6
	J[diagind(J)] .= 0
	J
end

# ╔═╡ d5e6d747-dfb4-418c-b00f-955d0de50994
let r = -1.5:0.1:1, 
	field_op = fieldterm(10, σx),
	J = chainJ(10),
	model = real.(symmetrize_op(xxzmodel(J, -0.73))),
	ψ0 = vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:10))))
	p = plot(;title="Test", legend=nothing)
	for h in r
		plot!(h .+ abs2.(eigen!(Hermitian(Matrix(model + 2h*field_operator))).vectors' * ψ0))
	end
	p
end

# ╔═╡ c64848bc-d819-40d3-b5cb-6b61b965ef1f
chainJ(10)

# ╔═╡ Cell order:
# ╠═5e92c53e-e948-11eb-17ce-5b1baf22cbf9
# ╠═bdab9d57-4c6e-4e55-8037-e29ec55c792c
# ╟─5493eab2-85ec-4805-9f2f-a6a447f9d16b
# ╟─d4fb5f55-37ac-49ab-b055-cb8a326cb3c9
# ╟─2b14057b-997b-45ad-a855-660165a55a1d
# ╠═749e5dde-3b8f-4b3d-95b1-427d270412df
# ╠═b1ee0e64-ce63-443d-b5ec-7448b9c20174
# ╠═f19e17d2-26da-4d15-bd23-d666ce0cb066
# ╠═68b9acb2-5cf9-4576-924e-2699397574ee
# ╠═66f9f163-c074-4e9e-941b-c35eec7a0dfb
# ╠═8d8d6f5d-6360-48f7-a298-fc75e8bed7db
# ╠═634fd7ef-dee7-4917-b237-ca59c4e537f4
# ╠═94418bc7-1a88-4eb5-a0bd-b11b55fdbfc6
# ╠═f74b4aeb-4899-4f9b-862a-b38d8dda6f31
# ╠═dfe5782a-fce2-409d-b7eb-1aabfdcaa18b
# ╠═20ab3428-48bb-4a93-90ad-c961cfea3b64
# ╠═c14bc300-4fec-4e49-89b1-d851c37da744
# ╠═a491cd5b-e610-4b8c-9a57-8e47494c9c6d
# ╠═d5e6d747-dfb4-418c-b00f-955d0de50994
# ╠═a208de1a-c167-42ed-af6c-1828a61f7d53
# ╠═c64848bc-d819-40d3-b5cb-6b61b965ef1f
