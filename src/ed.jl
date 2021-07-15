module ED

using ..Positions
using ..SimLib: logmsg, geometry_from_density

using Distributed
using JLD2
using LinearAlgebra
using Printf: @sprintf
using SharedArrays
using XXZNumerics

# simplify type definitions
const _FARRAY{N} = Array{Float64, N} where N

## Data structure

# index order
# [(spin,) eigen_state, shot, h, rho]
struct EDData
    geometry::Symbol
    dim::Int64
    α::Float64
    ρs::_FARRAY{1}
    fields::_FARRAY{1}
    eev::_FARRAY{5} # eigenstate expectation value of x magnetization ⟨i|Sⁱₓ|i⟩
    eon::_FARRAY{4} # eigenstate occupation number |⟨i|ψ⟩|²
    evals::_FARRAY{4} # energy eigenvalues Eᵢ
end

function EDData(geometry, dim, α, ρs, shots, system_size, fields)
    hilbert_space_dim = 2^(system_size-1) # due to symmetry
    hs = length(fields)
    
    EDData(
        geometry,
        dim,
        α,
        sort(ρs),
        fields,
        _FARRAY{5}(undef, system_size, hilbert_space_dim, shots, hs, length(ρs)),
        _FARRAY{4}(undef, hilbert_space_dim, shots, hs, length(ρs)),
        _FARRAY{4}(undef, hilbert_space_dim, shots, hs, length(ρs)),
    )
end

EDData(posdata::PositionData, α, fields) = EDData(Positions.geometry(posdata), Positions.dimension(posdata), α, Positions.ρs(posdata), Positions.shots(posdata), Positions.system_size(posdata), fields)

system_size(eddata::EDData) = size(eddata.eev, 1)
hilbert_space_dimension(eddata::EDData) = size(eddata.eev, 2)
shots(eddata::EDData) = size(eddata.eev, 3)
ρ_values(eddata::EDData) = eddata.ρs
fields(eddata::EDData) = eddata.fields
eev(eddata::EDData) = eddata.eev
eon(eddata::EDData) = eddata.eon
evals(eddata::EDData) = eddata.evals
α(eddata::EDData) = eddata.α
geometry(eddata::EDData) = eddata.geometry
dimension(eddata::EDData) = eddata.dim

## Saving/Loading
ed_datapath(prefix, geometry, N, dim, α) = joinpath(prefix, "data", @sprintf("ed_%s_%id_alpha_%.1f_N_%02i.jld2", geometry, dim, α, N))
ed_datapath(prefix, eddata::EDData) = ed_datapath(prefix, geometry(eddata), system_size(eddata), dimension(eddata), α(eddata))

function save(prefix, eddata::EDData)
    path = ed_datapath(prefix, eddata)
    mkpath(dirname(path))
    JLD2.jldsave(path; eddata)
end

load(path) = JLD2.load(path)["eddata"]
load(prefix, geometry, N, dim, α) = load(ed_datapath(prefix, geometry, N, dim, α))

## main function
function _ensemble_J_mean(interaction, geometry, positions)
    mapreduce(pos->sum(interaction_matrix(interaction, geometry, pos)), +, eachslice(positions; dims=3))/size(positions,2)/size(positions,3)
end

function _compute_core!(eev_out, evals_out, eon_out, model, field_values, field_operator, operators, ψ0)
    matrix = zeros(eltype(model), size(model)) # preallocate
    for (k, h) in enumerate(field_values)
        copyto!(matrix, model + h*field_operator) # this also converts from sparse to dense!
        evals, evecs = eigen!(Hermitian(matrix))
        for (l, evec) in enumerate(eachcol(evecs))
            for (op_index, op) in enumerate(operators)
                eev_out[op_index, l, k] = real(dot(evec, op, evec)) # dot conjugates the first arg
            end
        end
        evals_out[:, k] = evals
        eon_out[:, k] .= abs2.(evecs' * ψ0)
    end
end

run_ed(posdata::PositionData, α, fields; scale_field=:ensemble) = run_ed!(EDData(posdata, α, fields), posdata; scale_field)

function run_ed!(eddata::EDData, posdata::PositionData; scale_field=:ensemble)
    N = system_size(eddata)
    dim = Positions.dimension(posdata)
    ρs = ρ_values(eddata)
    field_values = fields(eddata)
    nshots = shots(eddata)
    interaction = PowerLaw(α(eddata))
    spin_ops = symmetrize_op.(op_list(σx/2, N))
    field_operator = sum(spin_ops)
    ψ0 = vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:N))))

    logmsg("ToDo: rho=$ρs")
    logmsg("with $nshots realizations and  $(length(field_values)) field values")
    for (i, ρ) in enumerate(ρs)
        logmsg("rho = $ρ")
        geom = geometry_from_density(geometry(eddata), ρ, N, dim)
        ensemble_J_mean = 0
        if scale_field == :ensemble
            # compute the ensembles mean J
            ensemble_J_mean = _ensemble_J_mean(interaction, geom, data(posdata)[:,:,:,i])
            logmsg("Ensemble J mean for rho_$i=$ρ: $ensemble_J_mean")
        end
        Threads.@threads for shot in 1:nshots
            logmsg(@sprintf("%03i/%03i", shot, nshots))
            J = interaction_matrix(interaction, geom, data(posdata)[:,:,shot,i])
            model = symmetrize_op(xxzmodel(J, -0.73))
            normed_field_values = field_values
            if scale_field == :shot
                normed_field_values *= sum(J)/N
            elseif scale_field == :ensemble
                normed_field_values *= ensemble_J_mean
            end
            _compute_core!(
                view(eddata.eev, :,:,shot,:,i), view(eddata.evals, :,shot,:,i), view(eddata.eon, :,shot,:,i),
                model, normed_field_values, field_operator, spin_ops, ψ0)
            # for (k, h) in enumerate(normed_field_values)
            #     H = model + h*field_operator
            #     evals, evecs = eigen!(Hermitian(Matrix(H)))
            #     for (l, evec) in enumerate(eachcol(evecs))
            #         for (spin, op) in enumerate(spin_ops)
            #             eev(eddata)[spin, l, shot, k, i] = real(dot(evec, op, evec)) # dot conjugates the first arg
            #         end
            #     end
            #     eddata.evals[:,shot, k, i] = evals
            #     eddata.eon[:, shot, k, i] .= abs2.(evecs' * ψ0)
            # end
        end
    end
    eddata
end

function _compute_core_parallel!(msg, eev_out, evals_out, eon_out, model, field_values, field_operator, operators, ψ0)
    logmsg(msg)
    _compute_core!(eev_out, evals_out, eon_out, model, field_values, field_operator, operators, ψ0)
end

function run_ed_parallel(posdata::PositionData, α, fields; scale_field=:ensemble, processes=Int[])
    N = Positions.system_size(posdata)
    dim = Positions.dimension(posdata)
    ρs = Positions.ρs(posdata)
    nshots = Positions.shots(posdata)
    hilbert_space_dim = 2^(N-1)
    geometry = Positions.geometry(posdata)

    if length(processes) == 0
        processes = workers()        
    end
    
    next_worker = let nworkers = length(processes); worker_index = 0;
        function inner()
            next_index = mod(worker_index, nworkers)+1
            worker_index += 1
            processes[next_index]
        end
    end
    
    #TODO only share with specified workers in processes arg
    eev = SharedArray{Float64}((N, hilbert_space_dim, nshots, length(fields), length(ρs)))
    evals = SharedArray{Float64}((hilbert_space_dim, nshots, length(fields), length(ρs)))
    eon = SharedArray{Float64}((hilbert_space_dim, nshots, length(fields), length(ρs)))

    done = 0
    todo = 0

    interaction = PowerLaw(α)
    spin_ops = symmetrize_op.(op_list(σx/2, N))
    field_operator = sum(spin_ops)
    ψ0 = SharedArray(vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:N)))))

    logmsg("ToDo: rho=$ρs")
    logmsg("with $nshots realizations and  $(length(fields)) field values")
    @sync begin
        for (i, ρ) in enumerate(ρs)
            logmsg("rho = $ρ"; doflush=false) # flush yields so taks start running
            geom = geometry_from_density(geometry, ρ, N, dim)
            ensemble_J_mean = 0
            if scale_field == :ensemble
                # compute the ensembles mean J
                ensemble_J_mean = _ensemble_J_mean(interaction, geom, data(posdata)[:,:,:,i])
                logmsg("Ensemble J mean for rho_$i=$ρ: $ensemble_J_mean"; doflush=false)
            end
            for shot in 1:nshots
                todo += 1
                worker = next_worker()
                @async begin
                    J = interaction_matrix(interaction, geom, data(posdata)[:,:,shot,i])
                    model = symmetrize_op(xxzmodel(J, -0.73))
                    normed_field_values = fields
                    if scale_field == :shot
                        normed_field_values *= sum(J)/N
                    elseif scale_field == :ensemble
                        normed_field_values *= ensemble_J_mean
                    end
                    #logmsg(@sprintf("#rho =%2i - %03i/%03i on #%02i", i, shot, nshots, worker))
                    remotecall_wait(_compute_core!, worker,
                        view(eev, :,:,shot,:,i), view(evals, :,shot,:,i), view(eon, :,shot,:,i),
                        model, normed_field_values, field_operator, spin_ops, ψ0)
                    done += 1
                    logmsg(@sprintf("Done %03i/%03i - #rho =%2i - %03i/%03i on #%02i", done, todo, i, shot, nshots, worker))
                end
                #logmsg("Scheduled #$todo")
                # @spawnat worker _compute_core_parallel!(@sprintf("#rho =%2i - %03i/%03i on #%02i", i, shot, nshots, worker),
                #     view(eev, :,:,shot,:,i), view(evals, :,shot,:,i), view(eon, :,shot,:,i),
                #     model, normed_field_values, field_operator, spin_ops, ψ0)
            end
        end
        logmsg("All queued")
    end
    EDData(geometry, dim, α, ρs, fields, sdata(eev), sdata(eon), sdata(evals))
end


_flat_to_indices(index, nshots) = (div(index-1, nshots)+1, rem(index-1, nshots)+1)
function _chunk_flat(interactions)
    idx = indexpids(interactions)
    nchunks = length(procs(interactions))
    total = size(interactions, 3) * size(interactions, 4) # nshots*nρs
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    (splits[idx]+1):splits[idx+1]
end

function _compute_core_parallel2!(eev_out, evals_out, eon_out, interactions, field_values, field_operator, scale_field, operators, ψ0)
    matrix = zeros(eltype(field_operator), size(field_operator)) # preallocate
    N = size(interactions, 1)
    nshots = size(interactions, 3)
    logmsg("Range: $(_chunk_flat(interactions)) on #0$(indexpids(interactions))")
    for index in _chunk_flat(interactions)
        i, shot = _flat_to_indices(index, nshots)
        logmsg("$index -> $shot, $i")
        J = @view interactions[:,:, shot, i]
        model = real.(symmetrize_op(xxzmodel(J, -0.73)))
        normed_field_values = field_values
        if scale_field == :shot
            normed_field_values *= sum(J)/N
        elseif scale_field == :ensemble
            normed_field_values *= sum(@view interactions[:,:,:,i])/nshots/N
        end
        
        for (k, h) in enumerate(normed_field_values)
            copyto!(matrix, model + h*field_operator) # this also converts from sparse to dense!
            evals, evecs = eigen!(Hermitian(matrix))
            for (l, evec) in enumerate(eachcol(evecs))
                for (op_index, op) in enumerate(operators)
                    eev_out[op_index, l, shot, k, i] = real(dot(evec, op, evec)) # dot conjugates the first arg
                end
            end
            evals_out[:, shot, k, i] = evals
            eon_out[:, shot, k, i] .= abs2.(evecs' * ψ0) # this allocates the most right now!
        end
        logmsg(@sprintf("Done %03i - #rho =%2i - %03i/%03i on #%02i", index, i, shot, nshots, indexpids(interactions)))
    end
end

function run_ed_parallel2(posdata::PositionData, α, fields; scale_field=:ensemble, processes=Int[])
    N = Positions.system_size(posdata)
    dim = Positions.dimension(posdata)
    ρs = Positions.ρs(posdata)
    nshots = Positions.shots(posdata)
    hilbert_space_dim = 2^(N-1)
    geometry = Positions.geometry(posdata)

    if length(processes) == 0
        processes = workers()        
    end
    
    #TODO only share with specified workers in processes arg    
    eev = SharedArray{Float64}((N, hilbert_space_dim, nshots, length(fields), length(ρs)))
    evals = SharedArray{Float64}((hilbert_space_dim, nshots, length(fields), length(ρs)))
    eon = SharedArray{Float64}((hilbert_space_dim, nshots, length(fields), length(ρs)))

    interaction = PowerLaw(α)
    spin_ops = real.(symmetrize_op.(op_list(σx/2, N)))
    field_operator = sum(spin_ops)
    ψ0 = SharedArray(vec(symmetrize_state(foldl(⊗, ((up+down)/√2 for _ in 1:N)))))

    interactions = SharedArray{Float64}((N,N,nshots,length(ρs)))
    logmsg("Calculating interactions")
    for i in 1:length(ρs)
        geom = geometry_from_density(geometry, ρs[i], N, dim)
        for shot in 1:nshots
            interactions[:,:, shot, i] .= interaction_matrix(interaction, geom, data(posdata)[:,:,shot,i])
        end
    end

    logmsg("ToDo: rho=$ρs")
    logmsg("with $nshots realizations and  $(length(fields)) field values")
    @sync for p in processes
        @async remotecall_wait(_compute_core_parallel2!, p,
            eev, evals, eon, interactions, fields, field_operator, scale_field, spin_ops, ψ0)
    end
    EDData(geometry, dim, α, ρs, fields, sdata(eev), sdata(eon), sdata(evals))
end

end #module