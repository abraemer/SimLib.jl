import Pkg
Pkg.activate(dirname(@__FILE__))

using LinearAlgebra: I, normalize!
using Random
using SimLib
using SpinSymmetry
using XXZNumerics

### positiondata setup
location = SaveLocation()
N = 8
shots = 30
fields = [-0.2, -0.1, 0.1, 0.2]
alpha = 6
pdd = PositionDataDescriptor(:box_pbc, 1, N, shots, [0.5, 1.0, 1.5], location)

Random.seed!(5)
posdata = load_or_create(pdd)
basis = symmetrized_basis(N, Flip(N), 0)

### tasks
hopping_operator = 1/2 * real.(σx⊗σx + σy⊗σy)

evaltask = Energies()
eontask = eontask = EigenstateOccupation("xpol", symmetrize_state(normalize!(ones(2^N)), basis))
eevtask = OperatorDiagonal("xmag", symmetrize_operator(sum(op_list(σx/2, N))/N, basis))
eltask  = EigenstateLocality("sz", symmetrize_operator(single_spin_op(σz, 1, N), basis))
eltask2 = EigenstateLocality("sx", symmetrize_operator(single_spin_op(σx, 1, N), basis))
eltask3 = EigenstateLocality("hopping", symmetrize_operator(kron(hopping_operator, I(2^(N-2))), basis))
eltask4 = EigenstateLocality("xmag", symmetrize_operator(sum(op_list(σx/2, N))/N, basis))
lsrtask = LevelSpacingRatio()

tasks = [evaltask, eontask, eevtask, lsrtask, eltask, eltask2, eltask3, eltask4]

### THREADED RUN

edd = EDDataDescriptor(pdd, alpha, fields, :ensemble, basis)

edata = run_ed(edd, posdata, tasks)
save.(edata)

ensdd = EnsembleDataDescriptor(edd)
save(create(ensdd))