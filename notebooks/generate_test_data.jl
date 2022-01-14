import Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using LinearAlgebra
using Random
using SimLib
using SpinSymmetry
using Statistics
using XXZNumerics

### positiondata setup
location = SaveLocation()
N = 8
shots = 30
fields = [-0.2, -0.1, 0.1, 0.2]
alpha = 6
basis = symmetrized_basis(N, Flip(N), 0)

## generate positions
Random.seed!(5)
pdd = PositionDataDescriptor(:box_pbc, 1, N, shots, [0.5, 1.0, 1.5], location)
posdata = load_or_create(pdd)

model = RandomPositionsXXZWithXField(pdd, PowerLaw(alpha), fields, :ensemble, basis)

### make ED tasks
hopping_operator = 1/2 * real.(σx⊗σx + σy⊗σy)

evaltask = Energies()
eontask = EigenstateOccupation("xpol", symmetrize_state(normalize!(ones(2^N)), basis))
eevtask = OperatorDiagonal("xmag", symmetrize_operator(sum(op_list(σx/2, N))/N, basis))
eltask  = EigenstateLocality("szsz", symmetrize_operator(single_spin_op(σz, 1, N)*single_spin_op(σz, 2, N), basis))
eltask2 = EigenstateLocality("sx", symmetrize_operator(single_spin_op(σx, 1, N), basis))
eltask3 = EigenstateLocality("hopping", symmetrize_operator(kron(hopping_operator, I(2^(N-2))), basis))
eltask4 = EigenstateLocality("xmag", symmetrize_operator(sum(op_list(σx/2, N))/N, basis))
lsrtask = LevelSpacingRatio()
#hcetask = HalfChainEntropy()

tasks = [evaltask, eontask, eevtask, lsrtask, eltask, eltask2, eltask3, eltask4]#, hcetask]

### THREADED RUN

edd = EDDataDescriptor(model, Full(), location)
@time edata = run_ed(edd, tasks, Serial())

save.(edata) # save all data

LinearAlgebra.BLAS.set_num_threads(1)

@time edata2 = run_ed(edd, tasks, Threaded())
@time edata3 = run_ed(edd, tasks, Parallel(4))

compare(a,b) = sum(abs2, a.data - b.data)
compare2(a,b) = sum(abs2, mean(a.data;dims=(1,2,3)) - mean(b.data;dims=(1,2,3)))
@show compare.(edata , edata2)
@show compare.(edata2, edata3)
@show compare2.(edata , edata2)
@show compare2.(edata2, edata3)

## Ensemble data
ensdd = EnsembleDataDescriptor(edd)
save(create(ensdd))
