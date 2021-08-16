@testset "ed.jl" begin
    @show Base.load_path()
    location = SaveLocation(;prefix=PREFIX)

    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    edd = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2])

    @test edd.pathdata.prefix == location.prefix

    # test threaded
    workers() != [1] && rmprocs(workers())
    log1 = joinpath(PREFIX, "ed_threaded.log")
    edata1 = to_file(log1) do
        load_or_create(edd; dosave=false) # also creates the positions file
    end
    @test isfile(datapath(pdd))
    @test log_contains(log1, "THREADS")

    # test multi-process
    addprocs(4)
    @everywhere import Pkg
    @everywhere Pkg.activate("..")
    @everywhere using SimLib
    log2 = joinpath(PREFIX, "ed_processes.log")
    edata2 = to_file(log2) do
        load_or_create(edd) # this reads the position file
    end
    @test log_contains(log2, "PROCESSES")

    @test edata1.eev ≈ edata2.eev
    @test edata1.evals ≈ edata2.evals
    @test edata1.eon ≈ edata2.eon
end