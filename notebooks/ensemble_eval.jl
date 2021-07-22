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

# ╔═╡ f891fe3a-eab4-11eb-0030-ab0b3b6beadc
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics
	gr()
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ 6471e1b6-f937-4ed3-8c46-a9ef55716bb0
data = SimLib.Ensembles.load("/home/adrian/results/julia-cusp/", :noisy_chain_pbc, 14, 1, 6);

# ╔═╡ 9ea89257-949f-4fa4-8a9f-6973a432bf92
ensemble_data = dropdims(mean(data.data; dims=1); dims=1);
#[h, rho, ensemble]
# ensemble: 1=microcanonical, 2=canonical, 3=diag

# ╔═╡ 4f0681d6-ee4d-4779-8b57-206a1f3a8c1b
md"""
ρ = $(@bind rhoIndex Slider(1:length(data.ρs)))
"""

# ╔═╡ 5bb087c5-cfb0-4042-afa4-5c2d0edf5c38
begin
	plot(;xlabel="Field h", ylabel="Magnetization", 
		title="N=$(SimLib.Ensembles.system_size(data))")
	plot!(data.fields, ensemble_data[:,rhoIndex,1];label="micro")
	plot!(data.fields, ensemble_data[:,rhoIndex,2];label="canon")
	plot!(data.fields, ensemble_data[:,rhoIndex,3];label="diago")
end

# ╔═╡ Cell order:
# ╠═f891fe3a-eab4-11eb-0030-ab0b3b6beadc
# ╠═6471e1b6-f937-4ed3-8c46-a9ef55716bb0
# ╠═9ea89257-949f-4fa4-8a9f-6973a432bf92
# ╠═4f0681d6-ee4d-4779-8b57-206a1f3a8c1b
# ╠═5bb087c5-cfb0-4042-afa4-5c2d0edf5c38
