@testset "positions.jl" begin
    using SimLib

    PREFIX = tempname()
    location = SaveLocation(;prefix=PREFIX)
    pdd = PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2], location)
    file = position_datapath(pdd)
    @test file == position_datapath(PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2]), PREFIX)
    @test file == position_datapath(PositionDataDescriptor(:box_pbc, 1, 8, 100, [0.1,0.2]), location)
    # clean if exists for some reason
    if isfile(file)
        rm(file)
    end
    pd = load_or_create(pdd)
    @test isfile(file)
    rm(file)
    Positions.save(pd)
    @test isfile(file)
    pd2 = Positions.load(pdd)
    for f in fieldnames(typeof(pd))
        @test getproperty(pd2, f) == getproperty(pd, f)
    end
end