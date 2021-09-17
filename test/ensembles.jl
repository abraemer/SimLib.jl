@testset "ensembles.jl" begin

    print("\n\n ####### ensembles.jl #######\n\n")

    location = SaveLocation(;prefix=PREFIX)

    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    basis = symmetrized_basis(7, Flip(7), 0)
    edd = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2], :ensemble, basis; suffix="threaded")
    ensdd = EnsembleDataDescriptor(edd)

    ensdata = load_or_create(ensdd, location) # should load the ED data since previous test saved it, and save to file without suffix

    ensdata2 = load_ensemble(:box, 1, 7, 6, location) # load previously generated data
    ## No idea for good test
    # check access of ensembles ?
    @test ensdata.canonical == ensdata2[:,:,:,ENSEMBLE_INDICES[:canonical]]
    @test ensdata.microcanonical == ensdata2[:,:,:,ENSEMBLE_INDICES[:microcanonical]]
    @test ensdata.diagonal == ensdata2[:,:,:,ENSEMBLE_INDICES[:diagonal]]
    # check attribute forwarding
    @test ensdd.fields == edd.fields
    @test ensdata2.descriptor == ensdata.descriptor
end
