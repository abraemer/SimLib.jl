module LSR

import ..ED
using ..SimLib: logmsg, geometry_from_density, meandrop, stddrop

using Statistics
using Printf: @sprintf
import JLD2

export levelspacingratio, LSRData, LSRDataDescriptor

## Data structure

struct LSRDataDescriptor
    geometry::Symbol
    dim::Int64
    N::Int64
    α::Float64
    ρs::Vector{Float64}
    shots::Int64
    fields::Vector{Float64}
end

LSRDataDescriptor(eddata::ED.EDData) = LSRDataDescriptor(ED.geometry(eddata), ED.dimension(eddata), ED.system_size(eddata), ED.α(eddata), ED.ρ_values(eddata), ED.shots(eddata), ED.fields(eddata))

struct LSRData
    descriptor::LSRDataDescriptor
    # [dummy, shot, h, rho]
    data::Array{Float64,4}
end

#LSRData(d::LSRDataDescriptor) = LSRData(d, Array{Float64,5}(undef, shots, length(fields), length(ρs), 3))
LSRData(eddata::ED.EDData; center_region=1.0) = LSRData(LSRDataDescriptor(eddata), levelspacingratio(eddata.evals; center_region))

# forward properties to descriptor
Base.getproperty(lsr::LSRData, s::Symbol) = hasfield(LSRData, s) ? getfield(lsr, s) : getproperty(lsr.descriptor, s)


## Saving/Loading
DEFAULT_FOLDER = "lsr"
lsr_datapath(prefix, geometry, N, dim, α; folder=DEFAULT_FOLDER, suffix="") = joinpath(prefix, folder, @sprintf("lsr_%s_%id_alpha_%.1f_N_%02i%s.jld2", geometry, dim, α, N, suffix))
lsr_datapath(prefix, lsrdata::LSRData; folder=DEFAULT_FOLDER, suffix="") = lsr_datapath(prefix, lsrdata.geometry, lsrdata.N, lsrdata.dim, lsrdata.α; folder, suffix)

function save(prefix, lsrdata::LSRData; folder=DEFAULT_FOLDER, suffix="")
    path = lsr_datapath(prefix, lsrdata; folder, suffix)
    mkpath(dirname(path))
    JLD2.jldsave(path; lsrdata)
end

load(path) = JLD2.load(path)["lsrdata"]
load(prefix, geometry, N, dim, α; folder=DEFAULT_FOLDER, suffix="") = load(lsr_datapath(prefix, geometry, N, dim, α; folder, suffix))

function levelspacingratio(levels; center_region=1.0)
    sizes = size(levels)
    L = sizes[1]
    cutoff = floor(Int, (L*(1-center_region)/2))
    range = (1+cutoff)+2:L-cutoff
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

Statistics.mean(lsr::LSRData) = meandrop(lsr.data; dims=1)
Statistics.std(lsr::LSRData) = stddrop(lsr.data; dims=1)
end #module