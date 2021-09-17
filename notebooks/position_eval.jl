### A Pluto.jl notebook ###
# v0.16.0

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

# ╔═╡ 2d6947b6-9306-48e3-b0fc-631c551b3f13
location = SaveLocation()

# ╔═╡ e13e2d66-25e5-47a9-8243-f66c84eecc53
pdd = PositionDataDescriptor(:noisy_chain_pbc, 1, 6, 30, [0.5, 1.0, 1.5], location)

# ╔═╡ a6dfa177-2d3c-410e-95ae-3b6eb1656de9
begin
	posdata = nothing
	with_terminal() do
		posdata = load_or_create(pdd);
	end
end

# ╔═╡ 4e4e8c35-af74-4a83-a3a6-18be7de97120
@bind i Slider(1:10)

# ╔═╡ 774f28cb-a09d-441b-b770-b8005f23ca6c
begin
	scatter(dropdims(posdata[:,:,i,:];dims=1), 1:posdata.system_size)
end

# ╔═╡ 19fbe000-a5cc-4804-839e-2dcf2d4f8f9b
posdata[:,:,1,1]

# ╔═╡ Cell order:
# ╠═0eba536c-e941-11eb-153a-d7d86c5bab3d
# ╠═2d6947b6-9306-48e3-b0fc-631c551b3f13
# ╠═e13e2d66-25e5-47a9-8243-f66c84eecc53
# ╠═a6dfa177-2d3c-410e-95ae-3b6eb1656de9
# ╠═4e4e8c35-af74-4a83-a3a6-18be7de97120
# ╠═774f28cb-a09d-441b-b770-b8005f23ca6c
# ╠═19fbe000-a5cc-4804-839e-2dcf2d4f8f9b
