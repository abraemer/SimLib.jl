@testset "positions.jl" begin

    print("\n\n ####### positions.jl #######\n\n")

    location = SaveLocation(;prefix=PREFIX)
    pdd = PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2], location)
    file = datapath(pdd)
    @test file == datapath(PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2]); prefix=PREFIX)
    @test file == datapath(PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2]), location)
    @test file == datapath(PositionDataDescriptor(:box_pbc, 1, 8), location)
    # clean if exists for some reason
    if isfile(file)
        rm(file)
    end
    pd = load_or_create(pdd)
    @test isfile(file)
    rm(file)
    save(pd)
    @test isfile(file)
    pd2 = load(pdd)
    pd3 = load_positions(:box_pbc, 1, 8; prefix=PREFIX)
    @test pd3 !== nothing # just check whether load worked
    for f in fieldnames(typeof(pd))
        @test getproperty(pd2, f) == getproperty(pd, f)
    end
end
