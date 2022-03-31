mutable struct PRTaskSimple <: ED.EDTask
    data
end

mutable struct PRTask{RB} <: ED.EDTask
    referencebasistype
    hamiltonianbasis
    data
end

struct PRTaskPairs{M,P,DT} <: ED.EDTask
    pairingmethod::M
    referencebasis::SparseMatrixCSC{Float64,Int64}
    projection::P # could be I, but will mostly be a SparseMatrixCSC anyways
    data::DT
end

ParticipationRatio() = PRTaskSimple(nothing)
ParticipationRatio(hamiltonianbasis::SymmetrizedBasis, referencebasis::PRBasis) = PRTask{typeof(referencebasis)}(referencebasis, hamiltonianbasis,nothing)
ParticipationRatio(hamiltonianbasis, ::ZBasis) = PRTaskSimple(nothing)

function ED.initialize!(task::Union{PRTaskSimple,PRTask}, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size)
    return task
end

function ED.initialize_local(task::PRTask)
    referencebasis = construct_basis(task.referencebasistype)
    projection = transformationmatrix(task.hamiltonianbasis)
    return PRTaskPairs(task.referencebasistype.pairingmethod, projection*referencebasis*projection', projection, task.data)
end

function ED.compute_task!(task::PRTaskPairs, evals, evecs, inds...; additional_parameters)
    perm_mat = task.projection * permutation_matrix(task.pairingmethod, additional_parameters[1]) * task.projection'
    n = min(size(task.data,1), size(evecs, 2))
    tempstorage1 = zeros(size(evecs, 1))
    tempstorage2 = zeros(size(evecs, 1))
    for i in 1:n
        mul!(tempstorage1, perm_mat, evecs[:, i])
        mul!(tempstorage2, task.referencebasis, tempstorage1)
        task.data[i, inds...] = participation_ratio(tempstorage2)
    end
end

function ED.compute_task!(task::PRTaskSimple, evals, evecs, inds...; additional_parameters)
    n = min(size(task.data,1), size(evecs, 2))
    @views participation_ratio!(task.data[1:n, inds...], evecs[:, 1:n])
end

function ED.failed_task!(task::Union{PRTaskSimple,PRTaskPairs}, inds...)
    task.data[:, inds...] .= NaN64
end


function ED.assemble(task::PRTaskSimple, edd)
    PRData(PRDataDescriptor(ZBasis(), edd), sdata(task.data))
end

function ED.assemble(task::PRTask, edd)
    PRData(PRDataDescriptor(task.referencebasistype, edd), sdata(task.data))
end
