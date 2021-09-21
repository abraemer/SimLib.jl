module EL

import ..ED
using .. SimLib
using ..SimLib: Maybe
using LinearAlgebra
using SpinSymmetry: basissize
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

struct ELData{T} <: ED.EDDerivedData
    descriptor::ELDataDescriptor{T}
    data::Array{Float64, 4}
end

ED._default_folder(::ELDataDescriptor) = "locality"
ED._filename_addition(eldd::ELDataDescriptor) = "_" * eldd.operatorname

load_el(geometry, dimension, system_size, α, operatorname, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(ELDataDescriptor(operatorname, geometry, dimension, system_size, α; prefix, suffix))

## functions

eigenstatelocality(eigen, operator) = eigenstatelocality!(Vector{Float64}(undef, length(eigen.values)-1), eigen, operator)

function eigenstatelocality!(res, eigen, operator)
    evals = eigen.values
    evecs = eigen.vectors
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


function ED.initialize!(task::ELTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, basissize(edd.basis)-1, edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::ELTask, ρindex, shot, fieldindex, eigen)
    eigenstatelocality!(view(task.data, :, shot, fieldindex, ρindex), eigen, task.operator)
end

function ED.failed_task!(task::ELTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::ELTask, edd)
    ELData(ELDataDescriptor(task.operator, task.operatorname, edd), sdata(task.data))
end

end #module
