
mutable struct RandomPositionsXXZ <: Model
    posdd::PositionDataDescriptor
    interaction
    scale_interactions
    Δ::Float64
    basis
end

#
# Constructors
#

"""
    parameter_dims(model) -> (dim1, dim2, ...)
"""
function parameter_dims(xxz::RandomPositionsXXZ)
    (xxz.posdd.shots, xxz.posdd.ρs)
end


"""
    initialize_model!(model, array_initializer)
"""
function initialize_model!(xxz::RandomPositionsXXZ, array_initializer)
    posdata = load(xxz.posdd)
    N = xxz.posdd.system_size
    other_params = parameter_dims(xxz)
    interactions = array_initializer(Float64, N,N,other_params...)
    _compute_interactions!(interactions, posdata, xxz.interaction, xxz.scale_interactions)
    # TODO
end


"""
    split_workload(model, num_parts) -> [parameter_iterators...]

    parameter_iterator -> [(parameter, parameterIndex)...]
"""
function split_workload end#(model, num_parts(runmode))


"""
    construct_Hamiltonian(model, parameter) -> AbstractMatrix
"""
function construct_Hamiltonian(xxz::RandomPositionsXXZ, params)
    real.(symmetrize_operator(xxzmodel(J, xxz.Δ), xxz.basis))
end



_flat_to_indices(index, nshots) = fldmod1(index, nshots)

function _workload_chunks(total, nchunks)
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    [(splits[i]+1):splits[i+1] for i in 1:nchunks]
end

function _ensemble_J_mean(interaction, geometry, positions)
    mapreduce(pos->sum(interaction_matrix(interaction, geometry, pos)), +, eachslice(positions; dims=3))/size(positions,2)/size(positions,3)
end

_flat_to_indices(index, nshots) = fldmod1(index, nshots)

function _workload_chunks(total, nchunks)
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    [(splits[i]+1):splits[i+1] for i in 1:nchunks]
end

function _compute_interactions!(arr, posdata, interaction, scale_interactions)
    logmsg("Calculating interactions (scaling = $(scale_interactions))")
    nshots, N = posdata.shots, posdata.system_size
    for (i, ρ) in enumerate(posdata.ρs)
        geom = geometry_from_density(posdata.geometry, ρ, N, posdata.dimension)
        for shot in 1:nshots
            arr[:,:, shot, i] .= interaction_matrix(interaction, geom, posdata[:,:,shot,i])
            if scale_interactions == :shot
                arr[:,:, shot, i] ./= sum(view(arr, :,:, shot, i))/N
            end
        end
        if scale_interactions == :ensemble
            factor = sum(@view arr[:,:,:,i])/nshots/N
            logmsg("rho = $ρ -> J_mean = $factor")
            arr[:,:,:,i] ./= factor
        end
    end
end
