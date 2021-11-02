### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 4af33954-3bc9-11ec-0799-7570e35dcd61
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__))
	using SimLib, Plots, PlutoUI, Statistics, LinearAlgebra, LaTeXStrings, XXZNumerics, SpinSymmetry, LsqFit, SparseArrays
	gr()
	LinearAlgebra.BLAS.set_num_threads(8)
	#cd(joinpath(@__DIR__, ".."))
end

# ╔═╡ 46425991-0870-4b28-92c4-45307b1567c1
path = abspath(joinpath(@__DIR__, "../data/zero-field"))

# ╔═╡ 90d15710-4cb2-4021-978c-41e7fe33cbee
entdata = load_entropy(:box_pbc, 1, 12, 6, 5; prefix=path);

# ╔═╡ f36d56c2-6d96-4c84-b006-b35b2c5192aa
count(isnan, entdata.data), length(entdata.data)

# ╔═╡ 5c9c8eec-10d8-43bb-b680-0ff0704d34a8
plot(entdata.ρs, meandrop(entdata.data; dims=(1,2,3,4)))

# ╔═╡ be3b69f9-cdf3-4f7c-bd76-9fc3a9803385
posdata = load_positions(:box_pbc, 1, 12; prefix=path);

# ╔═╡ 5aefbe53-096a-41ef-8e32-83ee37be92ea
edd = EDDataDescriptor(posdata, 6, [0], :ensemble, symmetrized_basis(zbasis(12,5)))

# ╔═╡ 8c2b33f3-44d6-499c-90c7-377fa96eb30d
task = HalfChainEntropyZBlock(6)

# ╔═╡ 43e48e73-7f33-4a58-909c-631fc4ae9750
ED.initialize!(task, edd, ED._array_constructor);

# ╔═╡ f424dd93-d549-4687-bb29-262862fa14a4
task.data

# ╔═╡ 9de36247-b784-4bf9-8899-91e91808ea2f
ED.compute_task!(task, 1,1,1,[], rand(792,792))

# ╔═╡ Cell order:
# ╠═4af33954-3bc9-11ec-0799-7570e35dcd61
# ╠═46425991-0870-4b28-92c4-45307b1567c1
# ╠═90d15710-4cb2-4021-978c-41e7fe33cbee
# ╠═f36d56c2-6d96-4c84-b006-b35b2c5192aa
# ╠═5c9c8eec-10d8-43bb-b680-0ff0704d34a8
# ╠═be3b69f9-cdf3-4f7c-bd76-9fc3a9803385
# ╠═5aefbe53-096a-41ef-8e32-83ee37be92ea
# ╠═8c2b33f3-44d6-499c-90c7-377fa96eb30d
# ╠═43e48e73-7f33-4a58-909c-631fc4ae9750
# ╠═f424dd93-d549-4687-bb29-262862fa14a4
# ╠═9de36247-b784-4bf9-8899-91e91808ea2f
