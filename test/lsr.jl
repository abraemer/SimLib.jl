@testset "lsr.jl" begin

    using Statistics

    print("\n\n ####### lsr.jl #######\n\n")

    location = SaveLocation(;prefix=PREFIX)

    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    basis = symmetrized_basis(7, Flip(7), 0)
    edd = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2], :ensemble, basis; suffix="full_ed_threaded")
    lsrdd = LSRDataDescriptor(edd)

    lsrdata = load_or_create(lsrdd) # should load the ED data since previous test saved it

    ## No idea for good test
    # check access of ensembles ?
    @test mean(lsrdata) ≈ dropdims(mean(lsrdata.data; dims=1); dims=1)
    @test std(lsrdata; center=0.5) ≈ dropdims(std(center_region(lsrdata, 0.5); dims=1); dims=1)
    # check attribute forwarding
    @test lsrdd.fields == edd.fields

    lsrdata2 = load_lsr(:box, 1, 7, 6, location; suffix="full_ed_threaded")
    @test lsrdata2.descriptor == lsrdata.descriptor
    @test lsrdata2.data ≈ lsrdata.data
end
