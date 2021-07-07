@testset "positions.jl" begin
    using SimLib.Positions

    dataset = PositionData(:box_pbc, [0.5,0.9], 4, 3, 2)
    create_positions!(dataset)
end