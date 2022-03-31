module OPDiag

import ..ED
using .. SimLib
using ..SimLib: Maybe
using LinearAlgebra
using SharedArrays: sdata

export OperatorDiagonal, OPDiagDataDescriptor, OPDiagData, load_opdiag

### Descriptor

struct OPDiagDataDescriptor{T} <: ED.EDDerivedDataDescriptor
    operator::Maybe{T}
    opname::String
    derivedfrom::ED.EDDataDescriptor
end


OPDiagDataDescriptor(opname::String, args...; kwargs...) = OPDiagDataDescriptor(nothing, opname, args...; kwargs...)
OPDiagDataDescriptor(op, opname::String, args...; kwargs...) = OPDiagDataDescriptor(op, opname, EDDataDescriptor(args...; kwargs...))

### Data obj

struct OPDiagData{T} <: SimLib.AbstractSimpleData
    descriptor::OPDiagDataDescriptor
    data::T
end

ED._default_folder(::OPDiagDataDescriptor) = "opdiag"
ED._filename_addition(opdd::OPDiagDataDescriptor) = "_" * opdd.opname

"""
    load_opdiag(opname, edd)
    load_opdiag(opname, model[, diagtype][, location])
"""
load_opdiag(args...; kwargs...) = load(OPDiagDataDescriptor(args...; kwargs...))

### Task

mutable struct OPDiagTask{H, O} <: ED.EDTask
    opname::String
    op::O
    data
end

OperatorDiagonal(opname, operator) = OPDiagTask{Val(ishermitian(operator)), typeof(operator)}(opname, operator, nothing)

const HermitianOPDiagTask = OPDiagTask{Val(true)}

function ED.initialize!(task::OPDiagTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(ComplexF64, spectral_size)
end

function ED.initialize!(task::HermitianOPDiagTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size)
end

function ED.compute_task!(task::HermitianOPDiagTask, evals, evecs, inds...; additional_parameters)
    n = min(size(task.data,1), size(evecs, 2))
    for (i, vec) in enumerate(eachcol(evecs))
        i <= n || break
        task.data[i, inds...] = real(dot(vec, task.op, vec))
    end
end

function ED.compute_task!(task::OPDiagTask, evals, evecs, inds...; additional_parameters)
    n = min(size(task.data,1), size(evecs, 2))
    for (i, vec) in enumerate(eachcol(evecs))
        i <= n || break
        task.data[i, inds...] = dot(vec, task.op, vec)
    end
end

function ED.failed_task!(task::OPDiagTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::OPDiagTask, edd)
    OPDiagData(OPDiagDataDescriptor(task.op, task.opname, edd), sdata(task.data))
end

Base.summary(task::OPDiagTask) = string(typeof(task)) * "($(task.opname))"

end #module
