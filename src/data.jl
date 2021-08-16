#TODO use Base.@kwdef ?
mutable struct SaveLocation
    prefix::String
    suffix::String
    SaveLocation(prefix, suffix) = new(string(prefix), string(suffix))
end

SaveLocation(;prefix=path_prefix(), suffix="") = SaveLocation(prefix, suffix)
SaveLocation(prefix::AbstractString) = SaveLocation(prefix, "")
SaveLocation(sl::SaveLocation) = sl

"""
    abstract AbstractDataDescriptor

Supertype for all `DataDescriptor`s.
A `DataDescriptor` holds information about a set parameters that can be used to obtain the corresponding `Data`. Either by `load`ing or by `create`ing it.
"""
abstract type AbstractDataDescriptor end

"""
    _filename(descriptor)
Generate the middle part of the save location. A prefix will by prepended by `joinpath` and a suffix with filetype will be appended.
"""
function _filename end

"""
    _convert_legacy_data(Val(:oldname), legacydata)
If data from before the Great Refactoring is loaded, this function will be used to convert it to a new and shiny `Data` object.
The dispatch is on the key for the object used by `JLD2`, converted to a `Symbol`.
"""
function _convert_legacy_data end

"""
    abstract AbstractData

Supertype for simulation data. Subtypes consist of an appropriate `DataDescriptor` and the actual data.
The `DataDescriptor` object should be stored in a field called `descriptor` or else they need to overwrite the function `descriptor(::MyDataType)` to return the appropriate `DataDescriptor`.
"""
abstract type AbstractData end

descriptor(data::AbstractData) = data.descriptor

#forward properties to descriptor
Base.getproperty(data::AbstractData, s::Symbol) = hasfield(typeof(data), s) ? getfield(data, s) : getproperty(data.descriptor, s)

"""
    AbstractSimpleData <: AbstractData

Supertype `Data` types that only consist of a single array and the descriptor.
"""
abstract type AbstractSimpleData <: AbstractData end

data(data::AbstractSimpleData) = data.data
Base.getindex(sdata::AbstractSimpleData, inds...) = getindex(data(sdata), inds...)
Base.setindex!(sdata::AbstractSimpleData, args...) = setindex!(data(sdata), args...)


"""
    datapath(dataOrDesc)
    datapath(dataOrDesc, path)
    datapath(dataOrDesc, prefix, suffix)
    datapath(dataOrDesc[, savelocation]; prefix, suffix)

Construct the path to where the data for this object should be saved to/loaded from.
There are a few different ways the path can be constructed.
 1. If a path is specified directly, take that
 2. Take prefix/suffix kwargs and fill in missing information from either the directly specified `SaveLocation` or take the one from the descriptor
"""
datapath(data::AbstractData, args...; kwargs...) = datapath(descriptor(data), args...; kwargs...) # unwrap Data -> DataDescriptor
# directly specified path takes precedence
datapath(::AbstractDataDescriptor, path::AbstractString) = path
# else merge SaveLocation and keyword args (latter have precedence)
datapath(desc::AbstractDataDescriptor, pathdata::SaveLocation=desc.pathdata; prefix=pathdata.prefix, suffix=pathdata.suffix) = datapath(desc, prefix, suffix)
# construct path
function datapath(desc::AbstractDataDescriptor, prefix::AbstractString, suffix::AbstractString) 
    if length(suffix) > 0
        joinpath(prefix, "$(_filename(desc))-$(suffix).jld2")
    else
        joinpath(prefix, "$(_filename(desc)).jld2")
    end
end

"""
    create(descriptor)

Perform the calculation described by the `DataDescriptor`.
"""
function create end

"""
    save(data)
    save(data, path)
    save(data[, savelocation]]; prefix, suffix)

Save the `Data` to the specified location.
See [`datapath`](@ref) for documention on the different variants to specifiy a path.
"""
save(data::AbstractData, args...; kwargs...) = save(data, datapath(data, args...; kwargs...))

function save(data::AbstractData, path::AbstractString)
    dname = dirname(path)
    if !isdir(dname)
        logmsg("Save directory: $dname does not exists. Creating!")
        mkpath(dname)
    end
    logmsg("Saving file: $path")
    JLD2.jldsave(path; data)
end


"""
    load(descriptor)
    load(descriptor, path)
    load(descriptor[, savelocation]]; prefix, suffix)

Load the data described by the `DataDescriptor`.
See [`datapath`](@ref) for documention on the different variants to specifiy a path.
"""
load(desc::AbstractDataDescriptor, args...; kwargs...) = load(datapath(desc, args...; kwargs...))

function load(path::AbstractString; throwerror=true)
    if !isfile(path)
        if !throwerror
            return nothing
        else
            error("$(abspath(path)) does not exist!")
        end
    end
    data = JLD2.load(path)
    if haskey(data, "data")
        data["data"]
    else
        # legacy file # try converting
        logmsg("Found legacy data. Some parameters might not be accurately loaded!")
        name, entry = first(data)
        _convert_legacy_data(Val(Symbol(name)), entry)
    end
end


"""
    load_or_create(descriptor; dosave)
    load_or_create(descriptor, path; dosave)
    load_or_create(descriptor[, savelocation]; dosave, prefix, suffix)

First try to `load` the data. If the file is not present or its contents do not match the requirements of the `descriptor`
then `create` the data. 
If `save` is true, also `save` the data (possibly overwriting what was there). Default: true
"""
function load_or_create(desc::AbstractDataDescriptor, pathargs...; dosave=true, kwargs...)
    logmsg("load_or_create with spec $(desc)")
    p = datapath(desc, pathargs...; kwargs...)
    if !isfile(p)
        logmsg("No data found at $(p).")
    else
        logmsg("Found existing data for $(desc).")
        data = load(p)
        descriptor(data) == desc && return data
        logmsg("Loaded data does not fit requirements.")
    end
    data = create(desc)
    dosave && save(data, p)
    data
end