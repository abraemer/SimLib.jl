module ED

using ..Positions
using ..SimLib
using ..SimLib: Maybe, FArray

using Distributed
import JLD2
using LinearAlgebra
using Printf: @sprintf
using SharedArrays
using XXZNumerics

export EDDataDescriptor, EDData, run_ed, load_ed

## Data structure

"""
    struct EDDataDescriptor

The important bits of information needed to specify for performing exact diagonalization on disordered XXZ Heisenberg models:
 - geometry
 - dimension
 - system_size
 - α
 - shots
 - densities ρs
 - field strengths
 - scaling of field strengths
 - symmetries to respect
For `load`ing data only the first 4 fields are required
Can also be constructed from a `PositionDataDescriptor` by supplying a the missing bits (α, fields).
"""
struct EDDataDescriptor <: SimLib.AbstractDataDescriptor
    geometry::Symbol
    dimension::Int
    system_size::Int
    α::Float64
    shots::Maybe{Int}
    ρs::Maybe{FArray{1}}
    fields::Maybe{FArray{1}}
    scale_fields::Maybe{Symbol}
    basis::Maybe{<:XXZNumerics.Symmetry.AbstractBasis}
    pathdata::SaveLocation
    function EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, pathdata::SaveLocation)
        geometry ∈ SimLib.GEOMETRIES || error("Unknown geometry: $geom")
        scale_fields ∈ [:none, :ensemble, :shot]
        new(geometry, dimension, system_size, α, shots, ismissing(ρs) ? missing : unique!(sort(vec(ρs))), ismissing(fields) ? missing : unique!(sort(vec(fields))), scale_fields, basis, pathdata::SaveLocation)
    end
end

EDDataDescriptor(posdata::PositionDataDescriptor, α, fields=missing, scale_fields=missing, basis=SpinFlip(zbasis(posdata.system_size)); prefix=posdata.pathdata.prefix, suffix=posdata.pathdata.suffix) =
    EDDataDescriptor(posdata.geometry, posdata.dimension, posdata.system_size, α, posdata.shots, posdata.ρs, fields, scale_fields, basis, SaveLocation(prefix, suffix))

EDDataDescriptor(posdata::PositionData, args...; kwargs...) = EDDataDescriptor(descriptor(posdata), args...; kwargs...)

function EDDataDescriptor(geometry, dimension, system_size, α, shots=missing, ρs=missing, fields=missing, scale_fields=missing, basis=SpinFlip(zbasis(system_size)), savelocation=SaveLocation(); prefix=savelocation.prefix, suffix=savelocation.suffix)
    EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, SaveLocation(; prefix, suffix))
end

function Base.:(==)(d1::EDDataDescriptor, d2::EDDataDescriptor)
    all(getfield(d1, f) == getfield(d2, f) for f in [:geometry, :dimension, :α, :shots, :ρs, :fields, :scale_fields, :basis])
end

# index order
# [(spin,) eigen_state, shot, h, rho]
"""
    struct EDData

Stores the actual data for the positions specified by the descriptor [`EDDataDescriptor`](@ref).

# Fields
 - `eev` Eigenstate Expectation Values for chosen operators ⟨i|Oʲ|i⟩ (default: Sₓ for each spin)
 - `eon` Eigenstate Occupation Numbers |⟨i|ψ⟩|²
 - `evals` Energy Eigenvalues Eᵢ

# Index order
The indices mean:
 - operator index (only applicable for `eev` field)
 - index of the eigen_state (from 1 to `hilbert_space_dimension`)
 - shot number
 - field index
 - density ρ

The default save directory is "data".
"""
struct EDData <: SimLib.AbstractData
    descriptor::EDDataDescriptor
    eev::FArray{5} # eigenstate expectation value of x magnetization ⟨i|Sⁱₓ|i⟩
    eon::FArray{4} # eigenstate occupation number |⟨i|ψ⟩|²
    evals::FArray{4} # energy eigenvalues Eᵢ
end

EDData(args...; kwargs...) = EDData(EDDataDescriptor(args...; kwargs...))

function EDData(desc::EDDataDescriptor)
    hilbert_space_dim = basis_size(desc.basis)
    N = desc.system_size
    shots = desc.shots
    ρcount = length(desc.ρs)
    hs = length(desc.fields)

    EDData(
        desc,
        FArray{5}(undef, N, hilbert_space_dim, shots, hs, ρcount),
        FArray{4}(undef, hilbert_space_dim, shots, hs, ρcount),
        FArray{4}(undef, hilbert_space_dim, shots, hs, ρcount),
    )
end


## Saving/Loading
SimLib._filename(desc::EDDataDescriptor) = filename(desc.geometry, desc.dimension, desc.system_size, desc.α)
filename(geometry, dimension, system_size, α) = @sprintf("data/ed_%s_%id_alpha_%.1f_N_%02i", geometry, dimension, α, system_size)

load_ed(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(EDDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))

function _guess_basis(N, hilbert_space_dimension)
    if hilbert_space_dimension == 2^N
        zbasis(N)
    elseif hilbert_space_dimension == 2^(N-1)
        SpinFlip(zbasis(N))
    else
        b = zbasis(N, div(N-1,2))
        if basis_size(b) == hilbert_space_dimension
            b
        else
            logmsg("[WARN]Could not guess a matching basis for N=$N dim=$hilbert_space_dimension")
            missing
        end
    end
end

function SimLib._convert_legacy_data(::Val{:eddata}, legacydata)
    logmsg("[WARN]Guessing parameters while loading ED legacy data:")
    logmsg("[WARN]  scale_fields")

    eev = legacydata.eev
    eon = legacydata.eon
    evals = legacydata.evals
    N, hilbert_space_dimension, shots, _, _ = size(eev)
    basis = _guess_basis(N, hilbert_space_dimension)
    ρs = legacydata.ρs
    fields = legacydata.fields
    geom = legacydata.geometry
    dim = legacydata.dim
    α = legacydata.α

    savelocation = SaveLocation(prefix="")
    desc = EDDataDescriptor(geom, dim, N, α, shots, ρs, fields, missing, basis, savelocation)
    EDData(desc, eev, eon, evals)
end

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

run_ed(posdata::PositionData, args...; kwargs...) = run_ed!(EDDataDescriptor(posdata, args...; kwargs...), posdata)

_flat_to_indices(index, nshots) = fldmod1(index, nshots)

function _workload_chunks(total, nchunks)
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    [(splits[i]+1):splits[i+1] for i in 1:nchunks]
end

function _compute_parallel(edd, posdata)
    N = edd.system_size
    hilbert_space_dim = basis_size(edd.basis)
    nshots = edd.shots
    nfields = length(edd.fields)
    nρs = length(edd.ρs)
    eev   = SharedArray{Float64}((N, hilbert_space_dim, nshots, nfields, nρs))
    evals = SharedArray{Float64}(   (hilbert_space_dim, nshots, nfields, nρs))
    eon   = SharedArray{Float64}(   (hilbert_space_dim, nshots, nfields, nρs))

    interactions = SharedArray{Float64}((N,N,nshots,nρs))
    _compute_interactions!(interactions, edd, posdata)

    workloads = _workload_chunks(nshots*nρs, length(procs(interactions)))

    @sync for (i, p) in enumerate(procs(interactions))
        @async remotecall_wait(_compute_parallel_job!, p,
            eev, evals, eon, interactions, workloads[i], edd)
    end

    EDData(edd, sdata(eev), sdata(eon), sdata(evals))
end

function _compute_parallel_job!(eev, evals, eon, interactions, workload, desc)
    # build operators and initial state here
    # -> is it worth to just to this once and use SharedArrays?
    # probably not much of a difference as the opjects here are quite small in RAM
    N = desc.system_size
    spin_ops = real.(symmetrize_op.(Ref(desc.basis), op_list(σx/2, N)))
    field_operator = sum(spin_ops)
    ψ0 = vec(symmetrize_state(desc.basis, foldl(⊗, ((up+down)/√2 for _ in 1:N)))) # all up in x-direction

    logmsg("Range: $(workload) on #$(indexpids(interactions))")
    _compute_core!(eev, evals, eon, interactions, workload, spin_ops, desc.fields, field_operator, ψ0, desc.basis)
end

function _compute_threaded(edd, posdata)
    N = edd.system_size
    hilbert_space_dim = basis_size(edd.basis)
    nshots = edd.shots
    nfields = length(edd.fields)
    nρs = length(edd.ρs)
    eev   = FArray{5}(undef, (N, hilbert_space_dim, nshots, nfields, nρs))
    evals = FArray{4}(undef,    (hilbert_space_dim, nshots, nfields, nρs))
    eon   = FArray{4}(undef,    (hilbert_space_dim, nshots, nfields, nρs))

    spin_ops = real.(symmetrize_op.(Ref(edd.basis), op_list(σx/2, N)))
    field_operator = sum(spin_ops)
    ψ0 = vec(symmetrize_state(edd.basis, foldl(⊗, ((up+down)/√2 for _ in 1:N))))

    interactions = FArray{4}(undef, (N,N,nshots,nρs))
    _compute_interactions!(interactions, edd, posdata)

    # use one thread less as the main thread should not be used for work (I think)
    workers = max(1, Threads.nthreads()-1)
    workloads = _workload_chunks(nshots*nρs, workers)
    @sync for i in 1:workers
        Threads.@spawn _compute_core!(eev, evals, eon, interactions, workloads[i], spin_ops, edd.fields, field_operator, ψ0, edd.basis)
    end

    EDData(edd, eev, eon, evals)
end

@inline function _compute_core!(eev_out, evals_out, eon_out, interactions, workload, operators, field_values, field_operator, ψ0, symmetry)
    nshots = size(interactions, 3)
    matrix = zeros(eltype(field_operator), size(field_operator)) # preallocate, but DENSE
    vec = zeros(eltype(ψ0), size(ψ0))
    for index in workload
        i, shot = _flat_to_indices(index, nshots)
        J = @view interactions[:,:, shot, i]
        model = real.(symmetrize_op(symmetry, xxzmodel(J, -0.73)))

        for (k, h) in enumerate(field_values)
            copyto!(matrix, model + h*field_operator) # this also converts from sparse to dense!
            try
                evals, evecs = eigen!(Hermitian(matrix))
                for (l, evec) in enumerate(eachcol(evecs))
                    for (op_index, op) in enumerate(operators)
                        eev_out[op_index, l, shot, k, i] = real(dot(evec, op, evec)) # dot conjugates the first arg
                    end
                end
                evals_out[:, shot, k, i] = evals
                eon_out[:, shot, k, i] .= abs2.(mul!(vec, evecs', ψ0)) # this allocates the most right now!
            catch e;
                logmsg("Error occured for #field=$k shot=$shot #rho=$i: $e")
                display(stacktrace(catch_backtrace()))
                eev_out[:, :, shot, k, i] .= NaN
                evals_out[ :, shot, k, i] .= NaN
                eon_out[   :, shot, k, i] .= NaN

                continue
            end
        end
        logmsg(@sprintf("Done %03i - #rho =%2i - %03i/%03i", index, i, shot, nshots))
    end
end

function _compute_interactions!(arr, edd, posdata)
    logmsg("Calculating interactions (scaling = $(edd.scale_fields))")
    interaction = PowerLaw(edd.α)
    nshots, N = edd.shots, edd.system_size
    for (i, ρ) in enumerate(edd.ρs)
        geom = geometry_from_density(edd.geometry, ρ, N, edd.dimension)
        for shot in 1:nshots
            arr[:,:, shot, i] .= interaction_matrix(interaction, geom, posdata[:,:,shot,i])
            if edd.scale_fields == :shot
                arr[:,:, shot, i] ./= sum(view(arr, :,:, shot, i))/N
            end
        end
        if edd.scale_fields == :ensemble
            factor = sum(@view arr[:,:,:,i])/nshots/N
            logmsg("rho = $ρ -> J_mean = $factor")
            arr[:,:,:,i] ./= factor
        end
    end
end

run_ed(desc::EDDataDescriptor) = create(desc)
function run_ed(desc::EDDataDescriptor, posdata::PositionData)
    # decide on mode of operation
    logmsg("ToDo: rho=$(desc.ρs)")
    logmsg("with $(desc.shots) realizations and  $(length(desc.fields)) field values")
    wcount = length(workers())
    if wcount > 1
        logmsg("Running ED on $(wcount) PROCESSES")
        _compute_parallel(desc, posdata)
    else
        logmsg("Running ED on $(Threads.nthreads()) THREADS")
        _compute_threaded(desc, posdata)
    end
end

function SimLib.create(desc::EDDataDescriptor)
    logmsg("Creating ED data for $(desc)")
    pdd = PositionDataDescriptor(desc.geometry, desc.dimension, desc.system_size, desc.shots, desc.ρs, desc.pathdata)
    logmsg("Needed Position data $(pdd)")
    posdata = load_or_create(pdd)
    run_ed(desc, posdata)
end

end #module
