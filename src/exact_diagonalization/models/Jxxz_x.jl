
mutable struct RandomPositionsXXZWithXField <: Model
    posdatadescriptor
    interaction
    fields
    scale_interactions
    basis
end

# small constructor - useful for loading
RandomPositionsXXZWithXField(posdatadescriptor, interaction) = RandomPositionsXXZWithXField(posdatadescriptor, interaction, nothing, nothing, nothing)

function Base.getproperty(xxz::RandomPositionsXXZWithXField, p::Symbol)
    if hasfield(typeof(xxz), p)
        return getfield(xxz, p)
    else
        return getproperty(getfield(xxz, :posdatadescriptor), p)
    end
end

model_name(::RandomPositionsXXZWithXField) = "Jxxz_with_field"
model_fileprefix(xxz::RandomPositionsXXZWithXField) = @sprintf("%s_%id_alpha_%.1f_N_%02i", xxz.geometry, xxz.dimension, xxz.interaction.α, xxz.system_size)

struct PreparedJXXZWithField{JT,SB}
    J::JT
    fields::Vector{Float64}
    basis::SB
end

# shot, field, rho
parameter_dims(model::PreparedJXXZWithField) = (size(model.J)[3], length(model.fields), size(model.J)[4])

function initialize_model!(model::RandomPositionsXXZWithXField, array_initializer)
    posdata = load(model.posdatadescriptor)
    shots = posdata.shots
    N = posdata.system_size
    ρs = posdata.ρs
    J = fill!(array_initializer(Float64,N,N,shots, length(ρs)), 0)
    _compute_interactions!(J, posdata, model.interaction, model.scale_interactions)

    return PreparedJXXZWithField(J, model.fields, model.basis)
end

function _ensemble_J_mean(interaction, geometry, positions)
    mapreduce(pos->sum(interaction_matrix(interaction, geometry, pos)), +, eachslice(positions; dims=3))/size(positions,2)/size(positions,3)
end

function _compute_interactions!(arr, posdata, interaction, scaling)
    logmsg("Calculating interactions (scaling = $(scaling))")
    N = size(arr, 1)
    nshots = size(arr, 3)
    for (i, ρ) in enumerate(posdata.ρs)
        geom = geometry_from_density(posdata.geometry, ρ, N, posdata.dimension)
        for shot in 1:nshots
            arr[:,:, shot, i] .= interaction_matrix(interaction, geom, posdata[:,:,shot,i])
            if scaling == :shot
                arr[:,:, shot, i] ./= sum(view(arr, :,:, shot, i))/N
            end
        end
        if scaling == :ensemble
            factor = sum(@view arr[:,:,:,i])/nshots/N
            logmsg("rho = $ρ -> J_mean = $factor")
            arr[:,:,:,i] ./= factor
        end
    end
end


function _workload_chunks(total, nchunks)
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    [(splits[i]+1):splits[i+1] for i in 1:nchunks]
end

function split_workload(model::PreparedJXXZWithField, parts)
    nshots = size(model.J, 3)
    nρs = size(model.J, 4)
    return _workload_chunks(nshots*nρs, parts)
end

function do_parameters(diag_callback, model::PreparedJXXZWithField, parameter_chunk)
    N = size(model.J, 1)
    spin_ops = real.(symmetrize_operator.(op_list(σx/2, N), Ref(model.basis)))
    field_operator = sum(spin_ops)
    nshots = size(model.J, 3)
    for index in parameter_chunk
        logmsg("Starting index $(index) of $(parameter_chunk)")
        ρindex, shot = _flat_to_indices(index, nshots)

        J = @view model.J[:,:, shot, ρindex]
        H_int = symmetrize_operator(xxzmodel(J, -0.73), model.basis)

        for (k, h) in enumerate(model.fields)
            logmsg("Doing #rho=$ρindex #shot=$shot #field=$k")
            parameterI = CartesianIndex(shot, k, ρindex)
            diag_callback(parameterI, H_int + h*field_operator, J)
        end
    end
end
