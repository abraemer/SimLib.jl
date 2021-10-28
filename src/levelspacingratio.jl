module LSR

import ..ED
using ..Levels
using ..SimLib
using ..SimLib: FArray
using SpinSymmetry: basissize

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
struct LSRData <: ED.EDDerivedData
    descriptor::LSRDataDescriptor
    # [dummy, shot, h, rho]
    data::FArray{4}
end

LSRData(lsrdd::LSRDataDescriptor) = LSRData(lsrdd, FArray{4}(undef, basissize(lsrdd.basis), lsrdd.shots, length(lsrdd.fields), length(lsrdd.ρs)))

## Saving/Loading

ED._default_folder(::LSRDataDescriptor) = "lsr"

load_lsr(geometry, dimension, system_size, α, location=SaveLocation(); prefix=location.prefix, suffix=location.suffix) = load(LSRDataDescriptor(geometry, dimension, system_size, α; prefix, suffix))

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
            if a ≈ b || b ≈ c || a ≈ c
                # prevent NaNs or -Infs
                # a=b=c -> NaN
                # a>b=c -> -Inf # this can happen if all of them are very close
                res[i, I] = 0
            else
                ratio = abs((b-a)/(c-b)) # use abs to be sure
                res[i, I] = min(ratio, 1/ratio)
            end
        end
    end
    res
end

function SimLib.create(lsrdd::LSRDataDescriptor)
    leveldata = load_or_create(LevelDataDescriptor(lsrdd.derivedfrom))
    LSRData(lsrdd, levelspacingratio(leveldata.data))
end

Statistics.mean(lsr::LSRData; center=1.0) = meandrop(center_region(lsr, center); dims=1)
Statistics.std(lsr::LSRData; center=1.0) = stddrop(center_region(lsr, center); dims=1)


mutable struct LSRTask <: ED.EDTask
    data
end

LevelSpacingRatio() = LSRTask(nothing)

function ED.initialize!(task::LSRTask, edd, arrayconstructor)
    task.data = arrayconstructor(Float64, basissize(edd.basis)-2, edd.shots, length(edd.fields), length(edd.ρs))
end

function ED.compute_task!(task::LSRTask, ρindex, shot, fieldindex, eigen)
    task.data[:, shot, fieldindex, ρindex] .= levelspacingratio(eigen.values)
end

function ED.failed_task!(task::LSRTask, ρindex, shot, fieldindex)
    task.data[:, shot, fieldindex, ρindex] .= NaN64
end

function ED.assemble(task::LSRTask, edd)
    LSRData(LSRDataDescriptor(edd), task.data)
end


end #module
