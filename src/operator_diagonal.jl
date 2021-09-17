module OPDiag

import ..ED
using .. SimLib
using ..SimLib: Maybe
using LinearAlgebra
using SpinSymmetry: basissize
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

load_opdiag(geometry, dimension, system_size, α, opname, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(OPDiagData(opname, geometry, dimension, system_size, α; prefix, suffix))

### Task

mutable struct OPDiagTask{H, O} <: ED.EDTask
    opname::String
    op::O
    data
end

OperatorDiagonal(opname, operator) = OPDiagTask{Val(ishermitian(operator)), typeof(operator)}(opname, operator, nothing)

const HermitianOPDiagTask = OPDiagTask{Val(true)}

function ED.initialize!(task::OPDiagTask, edd, arrayconstructor)
    task.data = arrayconstructor(ComplexF64, basissize(edd.basis), length(edd.fields), edd.shots, length(edd.ρs))
end

function ED.initialize!(task::HermitianOPDiagTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, basissize(edd.basis), length(edd.fields), edd.shots, length(edd.ρs))
end

function ED.compute_task!(task::HermitianOPDiagTask, ρindex, shot, fieldindex, eigen)
    for (i, vec) in enumerate(eachcol(eigen.vectors))
        task.data[i, fieldindex, shot, ρindex] = real(dot(vec, task.op, vec))
    end
end

function ED.compute_task!(task::OPDiagTask, ρindex, shot, fieldindex, eigen)
    for (i, vec) in enumerate(eachcol(eigen.vectors))
        task.data[i, fieldindex, shot, ρindex] = dot(vec, task.op, vec)
    end
end

function ED.failed_task!(task::OPDiagTask, ρindex, shot, fieldindex)
    task.data[:, fieldindex, shot, ρindex] .= NaN64
end

function ED.assemble(task::OPDiagTask, edd)
    OPDiagData(OPDiagDataDescriptor(task.op, task.opname, edd), sdata(task.data))
end

end #module
