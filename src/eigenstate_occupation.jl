module EON

import ..ED
using .. SimLib
using ..SimLib: Maybe
using LinearAlgebra
using SpinSymmetry: basissize
using SharedArrays: sdata

export EigenstateOccupation, EONDataDescriptor, EONData, load_eon

### Descriptor

struct EONDataDescriptor{T} <: ED.EDDerivedDataDescriptor
    ψ::Maybe{T}
    statename::String
    derivedfrom::ED.EDDataDescriptor
end


EONDataDescriptor(statename::String, args...; kwargs...) = EONDataDescriptor(nothing, statename, args...; kwargs...)
EONDataDescriptor(state, statename::String, args...; kwargs...) = EONDataDescriptor(state, statename, EDDataDescriptor(args...; kwargs...))

### Data obj

struct EONData{T} <: ED.EDDerivedData
    descriptor::EONDataDescriptor{T}
    data::Array{ComplexF64, 4}
end

ED._default_folder(::EONDataDescriptor) = "eon"
ED._filename_addition(opdd::EONDataDescriptor) = "_" * opdd.statename

load_eon(geometry, dimension, system_size, α, statename, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(EONDataDescriptor(statename, geometry, dimension, system_size, α; prefix, suffix))

### Task

mutable struct EONTask{T} <: ED.EDTask
    statename::String
    state::T
    data
end

EigenstateOccupation(statename, state) = EONTask(statename, state, nothing)


function ED.initialize!(task::EONTask, edd, arrayconstructor)
    task.data = arrayconstructor(ComplexF64, basissize(edd.basis), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::EONTask, ρindex, shot, fieldindex, eigen)
    for (i, vec) in enumerate(eachcol(eigen.vectors))
        task.data[i, shot, fieldindex, ρindex] = dot(vec, task.state)
    end
end

function ED.failed_task!(task::EONTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::EONTask, edd)
    EONData(EONDataDescriptor(task.state, task.statename, edd), sdata(task.data))
end

Base.summary(task::EONTask) = string(typeof(task)) * "($(task.statename))"

end #module
