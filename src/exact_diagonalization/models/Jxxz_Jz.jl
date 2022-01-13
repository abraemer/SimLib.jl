
mutable struct RandomPositionsXXZWithDegeneracyLifted <: Model
    posdatadescriptor
    interaction
    δs
    basis
end

# small constructor - useful for loading
RandomPositionsXXZWithDegeneracyLifted(posdatadescriptor, interaction) = RandomPositionsXXZWithDegeneracyLifted(posdatadescriptor, interaction, nothing, nothing)

function Base.getproperty(xxz::RandomPositionsXXZWithDegeneracyLifted, p::Symbol)
    if hasfield(typeof(xxz), p)
        return getfield(xxz, p)
    else
        return getproperty(getfield(xxz, :posdatadescriptor), p)
    end
end

model_name(::RandomPositionsXXZWithDegeneracyLifted) = "Jxxz_with_degeneracy_lifted"
model_fileprefix(xxz::RandomPositionsXXZWithDegeneracyLifted) = @sprintf("%s_%id_alpha_%.1f_N_%02i_lifted", xxz.geometry, xxz.dimension, xxz.interaction.α, xxz.system_size)

struct PreparedJXXZWithDegeneracyLifted{JT,SB}
    J::JT
    δs::Vector{Float64}
    basis::SB
end

# shot, field, rho
parameter_dims(model::PreparedJXXZWithDegeneracyLifted) = (size(model.J)[3], length(model.δs), size(model.J)[4])

function initialize_model!(model::RandomPositionsXXZWithDegeneracyLifted, array_initializer)
    posdata = load(model.posdatadescriptor)
    shots = posdata.shots
    N = posdata.system_size
    ρs = posdata.ρs
    J = fill!(array_initializer(Float64,N,N,shots, length(ρs)), 0)
    # see Jxxz_x.jl
    _compute_interactions!(J, posdata, model.interaction, nothing)

    return PreparedJXXZWithDegeneracyLifted(J, model.δs, model.basis)
end

_flat_to_indices(index, nshots) = fldmod1(index, nshots)


function split_workload(model::PreparedJXXZWithDegeneracyLifted, parts)
    nshots = size(model.J, 3)
    nρs = size(model.J, 4)
    return _workload_chunks(nshots*nρs, parts)
end

function do_parameters(diag_callback, model::PreparedJXXZWithDegeneracyLifted, parameter_chunk)
    N = size(model.J, 1)
    nshots = size(model.J, 3)
    for index in parameter_chunk
       ρindex, shot = _flat_to_indices(index, nshots)

        J = @view model.J[:,:, shot, ρindex]
        H_int = symmetrize_operator(xxzmodel(J, -0.73), model.basis)

        for (k, δ) in enumerate(model.δs)
            parameterI = CartesianIndex(shot, k, ρindex)
            z_fields = dropdims(sum(J; dims=1); dims=1)
            diag_callback(parameterI, H_int + δ * symmetrize_operator(z_field(z_fields), model.basis))
        end
    end
end
