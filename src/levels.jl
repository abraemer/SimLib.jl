module Levels

import ..ED
using .. SimLib
using ..SimLib: FArray
using SharedArrays: sdata

export Energies, LevelDataDescriptor, LevelData, load_levels

### Descriptor

struct LevelDataDescriptor <: ED.EDDerivedDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end



LevelDataDescriptor(args...; kwargs...) = LevelDataDescriptor(EDDataDescriptor(args...; kwargs...))



### Data obj

struct LevelData{N} <: ED.EDDerivedData
    descriptor::LevelDataDescriptor
    data::FArray{N}
end

ED._default_folder(::LevelDataDescriptor) = "levels"
#_filename_addition(::LevelData) = "" # default

"""
    load_levels(edd)
    load_levels(model[, diagtype][, location])
"""
load_levels(args...; kwargs...) = load(LevelDataDescriptor(args...; kwargs...))

### Task

mutable struct LevelTask <: ED.EDTask
    data
end

Energies() = LevelTask(nothing)

function ED.initialize!(task::LevelTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size)
end

function ED.compute_task!(task::LevelTask, evals, evecs, inds...)
    task.data[1:length(evals), inds...] .= evals
end

function ED.failed_task!(task::LevelTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::LevelTask, edd)
    LevelData(LevelDataDescriptor(edd), sdata(task.data))
end

end # module
