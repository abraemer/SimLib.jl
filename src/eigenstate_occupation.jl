module EON

import ..ED
using ..SimLib
using ..SimLib: Maybe
using LinearAlgebra
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

struct EONData{T, N} <: SimLib.AbstractSimpleData
    descriptor::EONDataDescriptor{T}
    data::Array{Float64, N}
end

ED._default_folder(::EONDataDescriptor) = "eon"
ED._filename_addition(opdd::EONDataDescriptor) = "_" * opdd.statename

"""
    load_eon(statename, edd)
    load_eon(statename, model[, diagtype] [, location])
"""
load_eon(args...; kwargs...) = load(EONDataDescriptor(args...; kwargs...))

### Task

mutable struct EONTask{T} <: ED.EDTask
    statename::String
    state::T
    data
end

EigenstateOccupation(statename, state) = EONTask(statename, state, nothing)


function ED.initialize!(task::EONTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size)
end

function ED.compute_task!(task::EONTask, evals, evecs, inds...)
    for (i, vec) in enumerate(eachcol(evecs))
        task.data[i, inds...] = abs2(dot(vec, task.state))
    end
end

function ED.failed_task!(task::EONTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::EONTask, edd)
    EONData(EONDataDescriptor(task.state, task.statename, edd), sdata(task.data))
end

Base.summary(task::EONTask) = string(typeof(task)) * "($(task.statename))"

end #module
