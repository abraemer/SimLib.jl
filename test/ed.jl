@testset "ed.jl" begin

    using LinearAlgebra: normalize!
    using SpinSymmetry

    print("\n\n ####### ed.jl #######\n\n")

    ### positiondata setup
    location = SaveLocation(;prefix=PREFIX)
    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    posdata = load_or_create(pdd)
    basis = symmetrized_basis(7, Flip(7), 0)

    ### tasks
    evaltask = Energies()
    eontask = eontask = EigenstateOccupation("xpol", symmetrize_state(normalize!(ones(2^7)), basis))
    eevtask = OperatorDiagonal("xmag", symmetrize_operator(sum(op_list(σx/2, 7))/7, basis))
    eltask = EigenstateLocality("sz", symmetrize_operator(single_spin_op(σz, 1, 7), basis))
    lsrtask = LevelSpacingRatio()
    #entropytask = HalfChainEntropy()

    tasks = [evaltask, eontask, eevtask, lsrtask, eltask]#, entropytask]

    ### THREADED RUN
    # remove processes to run threaded
    workers() != [1] && rmprocs(workers())

    edd1 = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2], :ensemble, basis; suffix="threaded")

    @test edd1.pathdata.prefix == location.prefix

    log1 = joinpath(PREFIX, "ed_threaded.log")
    SimLib.ED.reset_stats()
    edata1 = to_file(log1) do
        run_ed(edd1, posdata, tasks)
    end
    SimLib.ED.show_stats()
    save.(edata1)
    for data in edata1
        @test isfile(datapath(data))
    end
    @test log_contains(log1, "THREADS")

    ### MULTIPROCESS RUN
    addprocs(4)
    @everywhere import Pkg
    @everywhere Pkg.activate("..")
    @everywhere using SimLib

    edd2 = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2], :ensemble, basis; suffix="procs")

    log2 = joinpath(PREFIX, "ed_processes.log")
    SimLib.ED.reset_stats()
    edata2 = to_file(log2) do
        run_ed(edd2, posdata, tasks) # this reads the position file
    end
    SimLib.ED.show_stats()
    save.(edata2)
    for data in edata2
        @test isfile(datapath(data))
    end
    @test log_contains(log2, "PROCESSES")

    ## check values
    for (data1, data2) in zip(edata1, edata2)
        @test data1.data ≈ data1.data
    end

    for task in tasks
        ED.initialize!(task, edd1, ED._array_constructor)
        t = ED.initialize_local(task)
        ED.failed_task!(t, 1, 1, 1)
    end
end
