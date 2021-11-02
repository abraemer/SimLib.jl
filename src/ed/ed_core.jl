
## Data structure

abstract type DiagonalizationType end

function _ed_size end

struct Full <: DiagonalizationType end
Base.:(==)(::Full, ::Full) = true

_ed_size(::Full, edd) = basissize(edd.basis)

struct Sparse <: DiagonalizationType
    σ::Float64
    count::Int
end
Base.:(==)(s1::Sparse, s2::Sparse) = s1.σ == s2.σ && s1.count == s2.count

## TODO think about tolerance?
## TODO check convergence?
function execute_diag!(sp::Sparse, mat)

    return evals, evecs
end
_ed_size(sp::Sparse, edd) = sp.count

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
    basis::Maybe{Union{SymmetrizedBasis, FullZBasis, ZBlockBasis}}
    diagtype::Maybe{DiagonalizationType}
    pathdata::SaveLocation
    function EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, diagtype, pathdata::SaveLocation)
        geometry ∈ SimLib.GEOMETRIES || error("Unknown geometry: $geom")
        scale_fields ∈ [:none, :ensemble, :shot]
        new(geometry, dimension, system_size, α, shots, ismissing(ρs) ? missing : unique!(sort(collect(ρs))), ismissing(fields) ? missing : unique!(sort(collect(fields))), scale_fields, basis, diagtype, pathdata::SaveLocation)
    end
end

_default_basis(N) = symmetrized_basis(N, Flip(N), 0)
# unpack PositionData
EDDataDescriptor(posdata::PositionData, args...; kwargs...) = EDDataDescriptor(descriptor(posdata), args...; kwargs...)

# handle construction from PositionDataDescriptor and possible location overrides
function EDDataDescriptor(posdata::PositionDataDescriptor, args...; pathdata=posdata.pathdata, prefix=pathdata.prefix, suffix=pathdata.suffix, kwargs...)
    EDDataDescriptor(posdata.geometry, posdata.dimension, posdata.system_size, args...; ρs=posdata.ρs, shots=posdata.shots, prefix, suffix, kwargs...)
end

# full kwargs constructor
function EDDataDescriptor(geometry, dimension, system_size, α; kwargs...)
    EDDataDescriptor(; geometry, dimension, system_size, α, kwargs...)
end

function EDDataDescriptor(geometry, dimension, system_size; kwargs...)
    EDDataDescriptor(; geometry, dimension, system_size, kwargs...)
end

# full constructor
function EDDataDescriptor(;geometry, dimension, system_size, α, shots=missing, ρs=missing, fields=missing, scale_fields=missing, basis=_default_basis(system_size), diagtype=Full(), pathdata=SaveLocation(), prefix=pathdata.prefix, suffix=pathdata.suffix)
    EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, diagtype, SaveLocation(prefix, suffix))
end

# old positional constructor
EDDataDescriptor(posdata::PositionDataDescriptor, α, fields=missing, scale_fields=missing, basis=_default_basis(posdata.system_size), diagtype=Full(); prefix=posdata.pathdata.prefix, suffix=posdata.pathdata.suffix) =
    EDDataDescriptor(posdata.geometry, posdata.dimension, posdata.system_size, α, posdata.shots, posdata.ρs, fields, scale_fields, basis, diagtype, SaveLocation(prefix, suffix))

# function EDDataDescriptor(geometry, dimension, system_size, α, shots=missing, ρs=missing, fields=missing, scale_fields=missing, basis=_default_basis(system_size), diagtype=Full(), savelocation=SaveLocation(); prefix=savelocation.prefix, suffix=savelocation.suffix)
#     EDDataDescriptor(geometry, dimension, system_size, α, shots, ρs, fields, scale_fields, basis, diagtype, SaveLocation(; prefix, suffix))
# end

function Base.:(==)(d1::EDDataDescriptor, d2::EDDataDescriptor)
    all(getfield(d1, f) == getfield(d2, f) for f in [:geometry, :dimension, :α, :shots, :ρs, :fields, :scale_fields, :basis])
end

ed_size(edd) = _ed_size(edd.diagtype, edd)
ed_size(eddd::EDDerivedDataDescriptor) = ed_size(eddd.derivedfrom) # technically unnecessary as properties are forwarded anyways

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

## main function
function _ensemble_J_mean(interaction, geometry, positions)
    mapreduce(pos->sum(interaction_matrix(interaction, geometry, pos)), +, eachslice(positions; dims=3))/size(positions,2)/size(positions,3)
end

_flat_to_indices(index, nshots) = fldmod1(index, nshots)

function _workload_chunks(total, nchunks)
    splits = [round(Int, s) for s in range(0; stop=total, length=nchunks+1)]
    [(splits[i]+1):splits[i+1] for i in 1:nchunks]
end

function _compute_parallel!(tasks, edd, posdata)
    N = edd.system_size
    nshots = edd.shots
    nρs = length(edd.ρs)
    for task in tasks
        initialize!(task, edd, _sharedarray_constructor)
    end

    interactions = SharedArray{Float64}((N,N,nshots,nρs))
    _compute_interactions!(interactions, edd, posdata)

    workloads = _workload_chunks(nshots*nρs, length(procs(interactions)))

    @sync for (i, p) in enumerate(procs(interactions))
        @async remotecall_wait(_compute_parallel_job!, p,
            tasks, interactions, workloads[i], edd)
    end
end

function _compute_parallel_job!(tasks, interactions, workload, desc)
    # build operators and initial state here
    # -> is it worth to just to this once and use SharedArrays?
    # probably not much of a difference as the opjects here are quite small in RAM
    spin_ops = real.(symmetrize_operator.(op_list(σx/2, desc.system_size), Ref(desc.basis)))
    field_operator = sum(spin_ops)

    tasks = initialize_local.(tasks)

    logmsg("Range: $(workload) on #$(indexpids(interactions))")
    _compute_core!(tasks, interactions, workload, desc.fields, field_operator, desc.basis, desc.diagtype)
end

function _compute_threaded!(tasks, edd, posdata)
    N = edd.system_size
    nshots = edd.shots
    nρs = length(edd.ρs)
    for task in tasks
        initialize!(task, edd, _array_constructor)
    end

    spin_ops = real.(symmetrize_operator.(op_list(σx/2, N), Ref(edd.basis)))
    field_operator = sum(spin_ops)

    interactions = FArray{4}(undef, (N,N,nshots,nρs))
    _compute_interactions!(interactions, edd, posdata)

    # use one thread less as the main thread should not be used for work (I think)
    workers = max(1, Threads.nthreads()-1)
    workloads = _workload_chunks(nshots*nρs, workers)
    @sync for i in 1:workers
        Threads.@spawn _compute_core!(tasks, interactions, workloads[i], edd.fields, field_operator, edd.basis, edd.diagtype)
    end
end

## TODO better design? less copy-paste?
function _compute_core!(tasks, interactions, workload, field_values, field_operator, symmetry, diagtype::Full)
    tasks = initialize_local.(tasks)
    nshots = size(interactions, 3)
    matrix = zeros(eltype(field_operator), size(field_operator)) # preallocate, but DENSE
    for index in workload
        i, shot = _flat_to_indices(index, nshots)
        J = @view interactions[:,:, shot, i]
        model = real.(symmetrize_operator(xxzmodel(J, -0.73), symmetry))

        for (k, h) in enumerate(field_values)
            copyto!(matrix, model + h*field_operator) # this also converts from sparse to dense!
            try
                evals, evecs = eigen!(Hermitian(matrix))
                compute_task!.(tasks, i, shot, k, Ref(evals), Ref(evecs))
            catch e;
                logmsg("Error occured for #field=$k shot=$shot #rho=$i: $e")
                display(stacktrace(catch_backtrace()))
                failed_task!.(tasks, i, shot, k)
                continue
            end
        end
        logmsg(@sprintf("Done %03i - #rho =%2i - %03i/%03i", index, i, shot, nshots))
    end
end

function _compute_core!(tasks, interactions, workload, field_values, field_operator, symmetry, diagtype::Sparse)
    tasks = initialize_local.(tasks)
    nshots = size(interactions, 3)
    nev = diagtype.count
    sigma = diagtype.σ
    for index in workload
        i, shot = _flat_to_indices(index, nshots)
        J = @view interactions[:,:, shot, i]
        model = real.(symmetrize_operator(xxzmodel(J, -0.73), symmetry))

        for (k, h) in enumerate(field_values)
            try
                evals, evecs, nconv, niter, _, _ = eigs(Hermitian(model + h*field_operator); nev, sigma, which=:LM, tol=0, check=2)
                logmsg("converged=",nconv,"/",count," | niter=",niter)
                compute_task!.(tasks, i, shot, k, Ref(evals), Ref(evecs))
            catch e;
                logmsg("Error occured for #field=$k shot=$shot #rho=$i: $e")
                display(stacktrace(catch_backtrace()))
                failed_task!.(tasks, i, shot, k)
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



run_ed(desc::EDDataDescriptor, tasks::EDTask...) = run_ed(desc, tasks)

function run_ed(desc::EDDataDescriptor, tasks::Vector{EDTask})
    posdata = load_positions(desc.geometry, desc.dimension, desc.system_size, desc.pathdata)
    return run_ed(desc, posdata, tasks)
end

run_ed(desc::EDDataDescriptor, posdata::PositionData, tasks::EDTask...) = run_ed(desc, posdata, tasks)

function run_ed(desc::EDDataDescriptor, posdata::PositionData, tasks::Vector{EDTask})
    # decide on mode of operation
    logmsg("ToDo: rho=$(desc.ρs)")
    logmsg("with $(desc.shots) realizations and  $(length(desc.fields)) field values")
    wcount = length(workers())
    if wcount > 1
        logmsg("Running ED on $(wcount) PROCESSES")
        _compute_parallel!(tasks, desc, posdata)
    else
        logmsg("Running ED on $(Threads.nthreads()) THREADS")
        _compute_threaded!(tasks, desc, posdata)
    end
    return assemble.(tasks, Ref(desc))
end
