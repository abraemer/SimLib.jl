@testset "ed.jl" begin

    using LinearAlgebra: normalize!
    using SpinSymmetry

    print("\n\n ####### ed.jl #######\n\n")

    ### positiondata setup
    location = SaveLocation(;prefix=PREFIX)
    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    posdata = load_or_create(pdd)
    basis = symmetrized_basis(7, Flip(7), 0)
    hopping = real(single_spin_op(σx, 1, 7)*single_spin_op(σx, 2, 7) + single_spin_op(σy, 1, 7)*single_spin_op(σy, 2, 7))

    ### tasks
    evaltask = Energies()
    eontask = EigenstateOccupation("xpol", symmetrize_state(normalize!(ones(2^7)), basis))
    eevtask = OperatorDiagonal("xmag", symmetrize_operator(sum(op_list(σx/2, 7))/7, basis))
    eltask = EigenstateLocality("hopping", symmetrize_operator(hopping, basis))
    lsrtask = LevelSpacingRatio()
    iprtask = InverseParticipationRatio()

    tasks = [evaltask, eontask, eevtask, lsrtask, eltask, iprtask]
    @show isa.(tasks, ED.EDTask)

    model = RandomPositionsXXZWithXField(pdd, PowerLaw(6), [0], :ensemble, basis)

    # try all tasks once
    edd = EDDataDescriptor(model, Full(), location)
    for task in tasks
        ED.initialize!(task, ED._array_constructor, basissize(basis))
        t = ED.initialize_local(task)
        ED.failed_task!(task, 1)
        ED.assemble(task, edd)
    end

    model = RandomPositionsXXZWithXField(pdd, PowerLaw(6), [-0.2, -0.1, 0.1, 0.2], :ensemble, basis)

    @testset "$name" for (name, diagtype) in [("full_ed", Full()), ("shiftinvert", Sparse(ShiftInvert(), [0.25,0.75], 10, true))]
        ### THREADED RUN
        # remove processes to run threaded
        workers() != [1] && rmprocs(workers())

        edd1 = EDDataDescriptor(model, diagtype, location; suffix="$(name)_threaded")

        @test edd1.pathdata.prefix == location.prefix

        log1 = joinpath(PREFIX, "$(name)_threaded.log")
        edata1 = to_file(log1) do
            run_ed(edd1, tasks, Threaded())
        end

        save.(edata1)
        for data in edata1
            @test isfile(datapath(data))
        end
        @test log_contains(log1, "thread")

        ### MULTIPROCESS RUN

        edd2 = EDDataDescriptor(model, diagtype, location; suffix="$(name)_procs")

        log2 = joinpath(PREFIX, "$(name)_processes.log")
        edata2 = to_file(log2) do
            run_ed(edd2, tasks, Parallel(4)) # this reads the position file
        end
        save.(edata2)
        for data in edata2
            @test isfile(datapath(data))
        end
        @test log_contains(log2, "From worker")

        ## check values
        for (data1, data2) in zip(edata1, edata2)
            mysum(v) = sum(x -> isnan(x) ? 0 : abs(x), v)
            mymax(v) = maximum(x -> isnan(x) || isinf(x) ? -Inf : abs(x), v)
            @testset "$(typeof(data1))" begin
                @show typeof(data1)
                @show mysum(data1.data .- data2.data)
                @show mymax(data1.data .- data2.data)
                #@test data1.data ≈ data2.data nans=true atol=1e-10
            end
        end

        println("\n\nCheck logs at ",log1, " and ", log2, "\nContinue with <Return>")
        #readline(stdin)
        println("Continuing!")
    end

end
