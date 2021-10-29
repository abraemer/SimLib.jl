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

struct LevelData <: ED.EDDerivedData
    descriptor::LevelDataDescriptor
    data::FArray{4}
end

ED._default_folder(::LevelDataDescriptor) = "levels"
#_filename_addition(::LevelData) = "" # default

load_levels(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(LevelDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))

### Task

mutable struct LevelTask <: ED.EDTask
    data
end

Energies() = LevelTask(nothing)

function ED.initialize!(task::LevelTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, ED.ed_size(edd), edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::LevelTask, ρindex, shot, fieldindex, evals, evecs)
    task.data[1:length(evals), shot, fieldindex, ρindex] .= evals
end

function ED.failed_task!(task::LevelTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::LevelTask, edd)
    LevelData(LevelDataDescriptor(edd), sdata(task.data))
end

end # module
