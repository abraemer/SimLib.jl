@testset "ensembles.jl" begin

    print("\n\n ####### ensembles.jl #######\n\n")

    location = SaveLocation(;prefix=PREFIX)

    pdd = PositionDataDescriptor(:box, 1, 7, 20, [0.1, 0.2], location)
    edd = EDDataDescriptor(pdd, 6, [-0.2, -0.1, 0.1, 0.2], :ensemble, SpinFlip(zbasis(6)))
    ensdd = EnsembleDataDescriptor(edd)

    ensdata = SimLib.load_or_create(ensdd) # should load the ED data since previous test saved it

    ## No idea for good test
    # check access of ensembles ?
    @test ensdata.canonical == ensdata[:,:,:,ENSEMBLE_INDICES[:canonical]]
    @test ensdata.microcanonical == ensdata[:,:,:,ENSEMBLE_INDICES[:microcanonical]]
    @test ensdata.diagonal == ensdata[:,:,:,ENSEMBLE_INDICES[:diagonal]]
    # check attribute forwarding
    @test ensdd.fields == edd.fields
end