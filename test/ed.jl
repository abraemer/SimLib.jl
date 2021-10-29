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

    # try all tasks once
    edd = EDDataDescriptor(pdd; α=6, fields=[0])
    for task in tasks
        ED.initialize!(task, edd, ED._array_constructor)
        t = ED.initialize_local(task)
        ED.failed_task!(task, 1, 1, 1)
        ED.assemble(task, edd)
    end


    @testset "$name" for (name, diagtype) in [("full_ed", Full()), ("sparse_ed", Sparse(0,10))]
        ### THREADED RUN
        # remove processes to run threaded
        workers() != [1] && rmprocs(workers())

        edd1 = EDDataDescriptor(pdd; α=6, fields=[-0.2, -0.1, 0.1, 0.2], scale_fields=:ensemble, diagtype, basis, suffix="$(name)_threaded")

        @test edd1.pathdata.prefix == location.prefix

        log1 = joinpath(PREFIX, "$(name)_threaded.log")
        edata1 = to_file(log1) do
            run_ed(edd1, posdata, tasks)
        end

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

        edd2 = EDDataDescriptor(pdd; α=6, fields=[-0.2, -0.1, 0.1, 0.2], scale_fields=:ensemble, diagtype, basis, suffix="$(name)_procs")

        log2 = joinpath(PREFIX, "$(name)_processes.log")
        edata2 = to_file(log2) do
            run_ed(edd2, posdata, tasks) # this reads the position file
        end
        save.(edata2)
        for data in edata2
            @test isfile(datapath(data))
        end
        @test log_contains(log2, "PROCESSES")

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
        readline(stdin)
        println("Continuing!")
    end

end
