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

# ╔═╡ 0eba536c-e941-11eb-153a-d7d86c5bab3d
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI
	plotly()
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ a6dfa177-2d3c-410e-95ae-3b6eb1656de9
pd = SimLib.Positions.load(SimLib.path_prefix(), :noisy_chain_pbc, 6, 1);

# ╔═╡ 4e4e8c35-af74-4a83-a3a6-18be7de97120
@bind i PlutoUI.Slider(1:10)

# ╔═╡ 774f28cb-a09d-441b-b770-b8005f23ca6c
begin
	scatter(dropdims(pd[:,:,i,:];dims=1), 1:SimLib.Positions.system_size(pd))
end

# ╔═╡ 19fbe000-a5cc-4804-839e-2dcf2d4f8f9b
pd[:,:,1,1]

# ╔═╡ Cell order:
# ╠═0eba536c-e941-11eb-153a-d7d86c5bab3d
# ╠═a6dfa177-2d3c-410e-95ae-3b6eb1656de9
# ╠═4e4e8c35-af74-4a83-a3a6-18be7de97120
# ╠═774f28cb-a09d-441b-b770-b8005f23ca6c
# ╠═19fbe000-a5cc-4804-839e-2dcf2d4f8f9b
