### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f409dbe0-17b3-11ec-28ce-e78732e98c73
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics, SpinSymmetry
	gr()
	LinearAlgebra.BLAS.set_num_threads(8)
	cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ 55288508-762b-4087-bf3d-69f5ca43806c
RandomPositionsXXZWithXField(PositionDataDescriptor(:box_pbc, 1,8), PowerLaw(6), nothing, nothing, nothing)

# ╔═╡ dd6040ec-a6b2-425a-b75a-80bd9e4396e4
model = RandomPositionsXXZWithXField(PositionDataDescriptor(:box_pbc, 1,8), PowerLaw(6))

# ╔═╡ fbf8e37f-d687-4a20-8e12-1ef932d35483
eldata = load_el("xmag", model);

# ╔═╡ 7b111ece-3835-442c-aa72-940c3f4eef03
eldata.geometry

# ╔═╡ 08d7e059-dff7-460f-9926-ce1aa38f7915
md"""
#values: $(length(eldata.data))

#Inf   : $(sum(eldata.data .== -Inf))
"""

# ╔═╡ ae169b70-885b-4659-968c-4426f0535a54
size(eldata.data)

# ╔═╡ 989f3a23-8e50-4acc-8cda-1bdb8e79da64
means = meandrop(eldata.data; dims=(1,2,3));

# ╔═╡ 3daa32fc-78b3-43d5-bde6-0cfa24f444ed
md"""
Field: $(@bind fieldIndex Slider(1:length(eldata.fields)))

Density: $(@bind rhoIndex Slider(1:length(eldata.ρs)))
"""

# ╔═╡ 89ffd707-e5ef-485f-ba47-db07e3caa6f2
let p = plot(;legend=nothing, 
		title="ρ=$(eldata.ρs[rhoIndex]), h=$(eldata.fields[fieldIndex])")
	histogram!(p, vec(eldata.data[30:end-30,1,:,fieldIndex,rhoIndex]); bins=40)
end

# ╔═╡ eacbb47c-b86f-448c-b072-95e775f94e22
plot(eldata.fields, means; labels="ρ=" .* string.(eldata.ρs'))

# ╔═╡ 3eaa32b2-b41f-4609-8deb-9d69d80796f1
plot(eldata.ρs, means'; labels="h=" .* string.(eldata.fields'))

# ╔═╡ Cell order:
# ╠═f409dbe0-17b3-11ec-28ce-e78732e98c73
# ╠═55288508-762b-4087-bf3d-69f5ca43806c
# ╠═dd6040ec-a6b2-425a-b75a-80bd9e4396e4
# ╠═7b111ece-3835-442c-aa72-940c3f4eef03
# ╠═fbf8e37f-d687-4a20-8e12-1ef932d35483
# ╠═08d7e059-dff7-460f-9926-ce1aa38f7915
# ╠═ae169b70-885b-4659-968c-4426f0535a54
# ╠═989f3a23-8e50-4acc-8cda-1bdb8e79da64
# ╟─3daa32fc-78b3-43d5-bde6-0cfa24f444ed
# ╠═89ffd707-e5ef-485f-ba47-db07e3caa6f2
# ╠═eacbb47c-b86f-448c-b072-95e775f94e22
# ╠═3eaa32b2-b41f-4609-8deb-9d69d80796f1
