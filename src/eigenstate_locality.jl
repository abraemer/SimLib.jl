module EL

import ..ED
using ..SimLib
using ..SimLib: Maybe
using LinearAlgebra
using SharedArrays: sdata

export EigenstateLocality, ELDataDescriptor, ELData, load_el

### Descriptor

struct ELDataDescriptor{T} <: ED.EDDerivedDataDescriptor
    operator::Maybe{T}
    operatorname::String
    derivedfrom::ED.EDDataDescriptor
end

ELDataDescriptor(operatorname::String, args...; kwargs...) = ELDataDescriptor(nothing, operatorname, args...; kwargs...)
ELDataDescriptor(operator, operatorname::String, args...; kwargs...) = ELDataDescriptor(operator, operatorname, EDDataDescriptor(args...; kwargs...))

### Data obj

struct ELData{T, N} <: SimLib.AbstractSimpleData
    descriptor::ELDataDescriptor{T}
    data::Array{Float64, N}
end

ED._default_folder(::ELDataDescriptor) = "locality"
ED._filename_addition(eldd::ELDataDescriptor) = "_" * eldd.operatorname

"""
    load_el(operatorname, edd)
    load_el(operatorname, model, diagtype[, location])
"""
load_el(args...; kwargs...) = load(ELDataDescriptor(args...; kwargs...))

## functions

eigenstatelocality(eigen, operator) = eigenstatelocality!(Vector{Float64}(undef, length(eigen.values)-1), eigen, operator)

eigenstatelocality!(res, eigen, operator) = eigenstatelocality!(res, eigen.values, eigen.vectors, operator)

function eigenstatelocality!(res, evals, evecs, operator)
    energies = [eva + real(dot(eve, operator, eve)) for (eva, eve) in zip(evals, eachcol(evecs))]
    order = sortperm(energies)
    for (i, (index, nextindex)) in enumerate(zip(order[1:end-1], order[2:end]))
        res[i] = log(abs(dot(view(evecs, :, index), operator, view(evecs, :, nextindex))) / (energies[nextindex] - energies[index]))
    end
    res
end


### Task

mutable struct ELTask{T} <: ED.EDTask
    operatorname::String
    operator::T
    data
end

EigenstateLocality(operatorname, operator) = ELTask(operatorname, operator, nothing)


function ED.initialize!(task::ELTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size-1)
end

function ED.compute_task!(task::ELTask, evals, evecs, inds...)
    eigenstatelocality!(view(task.data, 1:length(evals)-1, inds...), evals, evecs, task.operator)
end

function ED.failed_task!(task::ELTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::ELTask, edd)
    ELData(ELDataDescriptor(task.operator, task.operatorname, edd), sdata(task.data))
end

Base.summary(task::ELTask) = string(typeof(task)) * "($(task.operatorname))"

end #module
