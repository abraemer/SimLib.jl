import Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using LinearAlgebra: I, normalize!
using Random
using SimLib
using SpinSymmetry
using XXZNumerics
using Statistics

import SimLib.ED

mutable struct TimeTask{T} <: SimLib.ED.EDTask
    elapsed::UInt64
    allocated::Int
    gctime::Int
    task::T
    TimeTask(task) = new{typeof(task)}(0,0,0,task)
end


SimLib.ED.initialize!(tt::TimeTask, arrayconstructor, spectral_size) = SimLib.ED.initialize!(tt.task, arrayconstructor, spectral_size)
# not thread/multiprocess safe!!!
SimLib.ED.initialize_local(tt::TimeTask) = (tt.task = SimLib.ED.initialize_local(tt.task); tt)
function SimLib.ED.compute_task!(tt::TimeTask, args...)
    gcbefore = Base.gc_num()
    timebefore = Base.time_ns()

    SimLib.ED.compute_task!(tt.task, args...)

    timeafter = Base.time_ns()
    gcdiff = Base.GC_Diff(Base.gc_num(), gcbefore)

    tt.elapsed += timeafter - timebefore
    tt.allocated += gcdiff.allocd
    tt.gctime += gcdiff.total_time
end
SimLib.ED.failed_task!(tt::TimeTask, args...) = SimLib.ED.failed_task!(tt.task, args...)

function SimLib.ED.assemble(tt::TimeTask, edd)
    println("Task time summary:\nElapsed: ", tt.elapsed/10e8, "s\nAllocated: ", tt.allocated, " bytes\nGC time: ", tt.gctime/10e8, "s\nTask: ", typeof(tt.task), "\n")
    SimLib.ED.assemble(tt.task, edd)
end

### positiondata setup
location = SaveLocation(prefix=joinpath(path_prefix(), "zero-field"))
N = 12
k = 5
shots = 30
fields = [0.0]
alpha = 6
basis = symmetrized_basis(N, k)

## generate positions
Random.seed!(5)
pdd = PositionDataDescriptor(:box_pbc, 1, N, shots, [0.5, 1.0, 1.5], location)
posdata = load_or_create(pdd)

## Model
model = RandomPositionsXXZWithXField(pdd, PowerLaw(alpha), fields, :ensemble, basis)

### make ED tasks
hopping_operator = 1/2 * real.(σx⊗σx + σy⊗σy)

evaltask = (Energies())
eontask = (EigenstateOccupation("xpol", symmetrize_state(normalize!(ones(2^N)), basis)))
eevtask = (OperatorDiagonal("xmag", symmetrize_operator(sum(op_list(σx/2, N))/N, basis)))
eltask  = (EigenstateLocality("sz", symmetrize_operator(single_spin_op(σz, 1, N), basis)))
eltask2 = (EigenstateLocality("szsz", symmetrize_operator(correlator(σz, 1, 2, N), basis)))
eltask3 = (EigenstateLocality("hopping", symmetrize_operator(kron(hopping_operator, I(2^(N-2))), basis)))
eltask4 = (EigenstateLocality("sxsx", symmetrize_operator(correlator(σx, 1, 2, N), basis)))
lsrtask = (LevelSpacingRatio())
hcetasks = [(HalfChainEntropyZBlock(;basis, L=i)) for i in 1:6]

tasks = [evaltask, eontask, eevtask, lsrtask, eltask, eltask2, eltask3, eltask4, hcetasks...]

### THREADED RUN

edd = EDDataDescriptor(model, Full(), location)

@time edata = run_ed(edd, tasks, Serial())

save.(edata) # save all data

@time edata2 = run_ed(edd, tasks, Threaded())
@time edata3 = run_ed(edd, tasks, Parallel(4))

compare(a,b) = sum(abs2, a.data - b.data)
compare2(a,b) = sum(abs2, mean(a.data;dims=(1,2,3)) - mean(b.data;dims=(1,2,3)))
@show compare.(edata , edata2)
@show compare.(edata2, edata3)
@show compare2.(edata , edata2)
@show compare2.(edata2, edata3)
