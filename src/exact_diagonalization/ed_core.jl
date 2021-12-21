

## main function

run_ed(descriptor, tasks...; runmode) = run_ed(descriptor, tasks, runmode)

function run_ed(descriptor, tasks, runmode)
    diagtype = descriptor.diagtype
    setup(runmode)
    array_type = array_initializer(runmode)

    model = initialize_model!(descriptor.model, array_type) # -> _compute_interactions
    task_array_init = (type, sizes...) -> array_type(type, sizes..., ed_size(diagtype), parameter_dims(model)...)
    initialize!.(tasks, task_array_init, spectral_size(diagtype, model), )
    parameter_chunks = split_workload(model, num_workers(runmode))
    start_computation(runmode, parameter_chunks) do chunk
        # call _compute_core! for each chunk
        _compute_core!(tasks, diagtype, model, chunk)
    end
    return assemble.(tasks, Ref(descriptor))
end

function _compute_core!(tasks, diagtype, model, parameter_chunk)
    tasks = initialize_local.(tasks)
    do_parameters(model, parameter_chunk) do parameter_index, H
        try
            logmsg("Starting index $(parameter_index) of $(parameter_chunk)")
            diagtime = 0.0
            tasktime = 0.0
            start = time()
            diagonalize!(H, diagtype) do diag_index, eigvals, eigvecs
                diagtime += time() - start
                start = time()
                compute_task!.(tasks, Ref(eigvals), Ref(eigvecs), Ref(diag_index), Ref(parameter_index))
                tasktime += time() - start
                start = time()
            end
            logmsg("Diagonalization took TDIAG=$(round(diagtime; digits=2))s and tasks took TTASK=$(round(tasktime; digits=2))s.")
        catch e;
            logmsg("Error during diagonalization occured for index $parameter_index : $e")
            display(stacktrace(catch_backtrace()))
            failed_task!.(tasks, :, Ref(parameterIndex))
        end
    end
end




# function _compute_parallel!(tasks, edd, posdata)
#     N = edd.system_size
#     nshots = edd.shots
#     nρs = length(edd.ρs)
#     for task in tasks
#         initialize!(task, edd, _sharedarray_constructor)
#     end

#     interactions = SharedArray{Float64}((N,N,nshots,nρs))
#     _compute_interactions!(interactions, edd, posdata)

#     workloads = _workload_chunks(nshots*nρs, length(procs(interactions)))

#     @sync for (i, p) in enumerate(procs(interactions))
#         @async remotecall_wait(_compute_parallel_job!, p,
#             tasks, interactions, workloads[i], edd)
#     end
# end

# function _compute_parallel_job!(tasks, interactions, workload, desc)
#     # build operators and initial state here
#     # -> is it worth to just to this once and use SharedArrays?
#     # probably not much of a difference as the opjects here are quite small in RAM
#     spin_ops = real.(symmetrize_operator.(op_list(σx/2, desc.system_size), Ref(desc.basis)))
#     field_operator = sum(spin_ops)

#     logmsg("Range: $(workload) on #$(indexpids(interactions))")
#     _compute_core!(tasks, interactions, workload, desc.fields, field_operator, desc.basis, desc.diagtype)
# end

# function _compute_threaded!(tasks, edd, posdata)
#     N = edd.system_size
#     nshots = edd.shots
#     nρs = length(edd.ρs)
#     for task in tasks
#         initialize!(task, edd, _array_constructor)
#     end

#     spin_ops = real.(symmetrize_operator.(op_list(σx/2, N), Ref(edd.basis)))
#     field_operator = sum(spin_ops)

#     interactions = FArray{4}(undef, (N,N,nshots,nρs))
#     _compute_interactions!(interactions, edd, posdata)

#     # use one thread less as the main thread should not be used for work (I think)
#     workers = max(1, Threads.nthreads()-1)
#     workloads = _workload_chunks(nshots*nρs, workers)
#     @sync for i in 1:workers
#         Threads.@spawn _compute_core!(tasks, interactions, workloads[i], edd.fields, field_operator, edd.basis, edd.diagtype)
#     end
# end

# ## TODO better design? less copy-paste?
# function _compute_core!(tasks, interactions, workload, field_values, field_operator, symmetry, diagtype::Full)
#     tasks = initialize_local.(tasks)
#     nshots = size(interactions, 3)
#     matrix = zeros(eltype(field_operator), size(field_operator)) # preallocate, but DENSE
#     for index in workload
#         i, shot = _flat_to_indices(index, nshots)
#         J = @view interactions[:,:, shot, i]
#         model = real.(symmetrize_operator(xxzmodel(J, -0.73), symmetry))

#         for (k, h) in enumerate(field_values)
#             copyto!(matrix, model + h*field_operator) # this also converts from sparse to dense!
#             try
#                 evals, evecs = eigen!(Hermitian(matrix))
#                 compute_task!.(tasks, i, shot, k, Ref(evals), Ref(evecs))
#             catch e;
#                 logmsg("Error occured for #field=$k shot=$shot #rho=$i: $e")
#                 display(stacktrace(catch_backtrace()))
#                 failed_task!.(tasks, i, shot, k)
#                 continue
#             end
#         end
#         logmsg(@sprintf("Done %03i - #rho =%2i - %03i/%03i", index, i, shot, nshots))
#     end
# end

# function _compute_core!(tasks, interactions, workload, field_values, field_operator, symmetry, diagtype::Sparse)
#     tasks = initialize_local.(tasks)
#     nshots = size(interactions, 3)
#     nev = diagtype.count
#     sigma = diagtype.σ
#     for index in workload
#         i, shot = _flat_to_indices(index, nshots)
#         J = @view interactions[:,:, shot, i]
#         model = real.(symmetrize_operator(xxzmodel(J, -0.73), symmetry))

#         for (k, h) in enumerate(field_values)
#             try
#                 evals, evecs, nconv, niter, _, _ = eigs(Hermitian(model + h*field_operator); nev, sigma, which=:LM, tol=0, check=2)
#                 evals, evecs = LinearAlgebra.sorteig!(evals, evecs)
#                 logmsg("converged=",nconv,"/",nev," | niter=",niter)
#                 compute_task!.(tasks, i, shot, k, Ref(evals), Ref(evecs))
#             catch e;
#                 logmsg("Error occured for #field=$k shot=$shot #rho=$i: $e")
#                 display(stacktrace(catch_backtrace()))
#                 failed_task!.(tasks, i, shot, k)
#                 continue
#             end
#         end
#         logmsg(@sprintf("Done %03i - #rho =%2i - %03i/%03i", index, i, shot, nshots))
#     end
# end




# run_ed(desc::EDDataDescriptor, tasks::EDTask...) = run_ed(desc, tasks)

# function run_ed(desc::EDDataDescriptor, tasks::Vector{EDTask})
#     posdata = load_positions(desc.geometry, desc.dimension, desc.system_size, desc.pathdata)
#     return run_ed(desc, posdata, tasks)
# end

# run_ed(desc::EDDataDescriptor, posdata::PositionData, tasks::EDTask...) = run_ed(desc, posdata, tasks)

# function run_ed(desc::EDDataDescriptor, posdata::PositionData, tasks::Vector{<:EDTask})
#     # decide on mode of operation
#     logmsg("ToDo: rho=$(desc.ρs)")
#     logmsg("with $(desc.shots) realizations and  $(length(desc.fields)) field values")
#     wcount = length(workers())
#     if wcount > 1
#         logmsg("Running ED on $(wcount) PROCESSES")
#         _compute_parallel!(tasks, desc, posdata)
#     else
#         logmsg("Running ED on $(Threads.nthreads()) THREADS")
#         _compute_threaded!(tasks, desc, posdata)
#     end
#     return assemble.(tasks, Ref(desc))
# end
