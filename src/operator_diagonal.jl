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

struct OPDiagData{T} <: ED.EDDerivedData
    descriptor::OPDiagDataDescriptor
    data::T
end

ED._default_folder(::OPDiagDataDescriptor) = "opdiag"
ED._filename_addition(opdd::OPDiagDataDescriptor) = "_" * opdd.opname

load_opdiag(geometry, dimension, system_size, α, opname, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(OPDiagDataDescriptor(opname, geometry, dimension, system_size, α; prefix, suffix))

### Task

mutable struct OPDiagTask{H, O} <: ED.EDTask
    opname::String
    op::O
    data
end

OperatorDiagonal(opname, operator) = OPDiagTask{Val(ishermitian(operator)), typeof(operator)}(opname, operator, nothing)

const HermitianOPDiagTask = OPDiagTask{Val(true)}

function ED.initialize!(task::OPDiagTask, edd, arrayconstructor)
    task.data = arrayconstructor(ComplexF64, ED.ed_size(edd), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.initialize!(task::HermitianOPDiagTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, ED.ed_size(edd), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::HermitianOPDiagTask, ρindex, shot, fieldindex, evals, evecs)
    for (i, vec) in enumerate(eachcol(evecs))
        task.data[i, shot, fieldindex, ρindex] = real(dot(vec, task.op, vec))
    end
end

function ED.compute_task!(task::OPDiagTask, ρindex, shot, fieldindex, evals, evecs)
    for (i, vec) in enumerate(eachcol(evecs))
        task.data[i, shot, fieldindex, ρindex] = dot(vec, task.op, vec)
    end
end

function ED.failed_task!(task::OPDiagTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::OPDiagTask, edd)
    OPDiagData(OPDiagDataDescriptor(task.op, task.opname, edd), sdata(task.data))
end

Base.summary(task::OPDiagTask) = string(typeof(task)) * "($(task.opname))"

end #module
