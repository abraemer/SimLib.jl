module LSR

import ..ED
using ..Levels
using ..SimLib
using ..SimLib: FArray
using SharedArrays: sdata

import Statistics

export levelspacingratio, LSRData, LSRDataDescriptor, center_region, load_lsr, LevelSpacingRatio

## Data structure

"""
    struct LSRDataDescriptor

Carries the information to construct a [`EDDataDescriptor`](!ref) object.
 - geometry
 - dimension
 - system_size
 - α
 - shots
 - densities ρs
 - field strengths
 - scaling of field strengths
 - symmetries to respect
For `load`ing data only the first 4 fields are required.
Can also be constructed from a `PositionDataDescriptor` by supplying a the missing bits (α, fields).
"""
struct LSRDataDescriptor <: ED.EDDerivedDataDescriptor
    derivedfrom::ED.EDDataDescriptor
end

LSRDataDescriptor(args...; kwargs...) = LSRDataDescriptor(EDDataDescriptor(args...; kwargs...))


"""
    struct LSRData

Stores the actual level-spacing ratios (LSR) specified by the descriptor [`LSRDataDescriptor`](@ref).
Level spacing ratio is defined as lsr_i = min(r_i, 1/r_i) where r_i = (E_(i-2) - E_(i-1))/(E_(i-1) - E_i)

# Index order
The indices mean:
 - i (as in lsr_i)
 - shot index
 - field index
 - density ρ

The default save directory is "lsr".

`Statistics.mean` and `Statistics.std` are overloaded to act on the first dimension to conveniently compute
mean LSR and its variance.
"""
struct LSRData{N} <: SimLib.AbstractSimpleData
    descriptor::LSRDataDescriptor
    # [dummy, shot, h, rho]
    data::FArray{N}
end

LSRData(lsrdd::LSRDataDescriptor) = LSRData(lsrdd, FArray{4}(undef, ED.ed_size(lsrdd), lsrdd.shots, length(lsrdd.fields), length(lsrdd.ρs)))

## Saving/Loading

ED._default_folder(::LSRDataDescriptor) = "lsr"

"""
    load_lsr(edd)
    load_lsr(model[, diagtype][, location])
"""
load_lsr(args...; kwargs...) = load(LSRDataDescriptor(args...; kwargs...))

function center_indices(L, center_region)
    cutoff = floor(Int, (L*(1-center_region)/2))
    (1+cutoff):L-cutoff
end

center_region(lsr::LSRData, center) = @view lsr.data[center_indices(size(lsr.data, 1), center), :, :, :]

function levelspacingratio(levels; center=1.0)
    sizes = size(levels)
    range = center_indices(sizes[1]-2, center) .+ 2
    res = Array{Float64, length(sizes)}(undef, length(range), sizes[2:end]...)
    for I in CartesianIndices(axes(levels)[2:end])
        for (i,j) in enumerate(range)
            a,b,c = levels[j-2,I], levels[j-1,I], levels[j,I]
            ratio = abs((b-a)/(c-b)) # use abs to be sure
            res[i, I] = min(ratio, 1/ratio)
        end
    end
    res
end

function SimLib.create(lsrdd::LSRDataDescriptor)
    leveldata = load_or_create(LevelDataDescriptor(lsrdd.derivedfrom))
    LSRData(lsrdd, levelspacingratio(leveldata.data))
end

mutable struct LSRTask <: ED.EDTask
    data
end

LevelSpacingRatio() = LSRTask(nothing)

function ED.initialize!(task::LSRTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, spectral_size-2)
end

function ED.compute_task!(task::LSRTask, evals, evecs, inds...; additional_parameters)
    n = min(size(task.data,1), length(evals)-2)
    task.data[1:n, inds...] .= levelspacingratio(view(evals, 1:n+2))
end

function ED.failed_task!(task::LSRTask, inds...)
    task.data[:, inds...] .= NaN64
end

function ED.assemble(task::LSRTask, edd)
    LSRData(LSRDataDescriptor(edd), sdata(task.data))
end

###
## Mean LSR
###

mutable struct MeanLSRTask <: ED.EDTask
    data
end

MeanLevelSpacingRatio() = MeanLSRTask(nothing)

function ED.initialize!(task::MeanLSRTask, arrayconstructor, spectral_size)
    task.data = arrayconstructor(Float64, 1)
end

function ED.compute_task!(task::MeanLSRTask, evals, evecs, inds...)
    task.data[1, inds...] = mean(levelspacingratio(evals))
end

function ED.failed_task!(task::MeanLSRTask, inds...)
    task.data[1, inds...] .= NaN64
end

function ED.assemble(task::MeanLSRTask, edd)
    LSRData(LSRDataDescriptor(edd), sdata(task.data))
end



end #module
