abstract type EDDerivedDataDescriptor <: SimLib.AbstractDataDescriptor end

function Base.:(==)(eddd1::EDDerivedDataDescriptor, eddd2::EDDerivedDataDescriptor)
    let d1 = eddd1.derivedfrom, d2 = eddd2.derivedfrom
        all(getfield(d1, f) == getfield(d2, f) for f in [:geometry, :dimension, :α, :shots, :ρs, :fields, :scale_fields, :basis])
    end
end

# simply forward all properties, that are directly part of the descriptor
function Base.getproperty(eddd::EDDerivedDataDescriptor, p::Symbol)
    if hasfield(typeof(eddd), p)
        return getfield(eddd, p)
    else
        return getproperty(getfield(eddd, :derivedfrom), p)
    end
end

abstract type EDDerivedData <: SimLib.AbstractSimpleData end

function _default_folder end
_filename_addition(::EDDerivedDataDescriptor) = ""

function Base.getproperty(eddd::EDDerivedData, s::Symbol)
    if hasfield(typeof(eddd), s)
        return getfield(eddd, s)
    else
        return getproperty(getfield(eddd, :descriptor), s)
    end
end

function SimLib._filename(eddd::EDDerivedDataDescriptor)
    return joinpath(
        _default_folder(eddd),
        filename_base(eddd.geometry, eddd.dimension, eddd.system_size, eddd.α) * _filename_addition(eddd))
end
filename_base(geometry, dim, N, α) = @sprintf("%s_%id_alpha_%.1f_N_%02i", geometry, dim, α, N)


abstract type EDTask end

# output arrays
function initialize! end

# can preallocate compute space
# Note: This needs to wrap the Task in an additional struct
# The additional memory may not be a part of the same struct
# otherwise the code breaks in a multithreaded (not multiprocessed) setting
# as all threads share the same memory location for their temporary results
initialize_local(task::EDTask) = task

# task, ρindex, shot, fieldindex, eigen
function compute_task! end

# task, ρindex, shot, fieldindex
function failed_task! end

# task, EDDataDescriptor -> Data object
function assemble end

_array_constructor(type, dims...) = zeros(type, dims)
_sharedarray_constructor(type, dims...) = SharedArray{type}(dims)
