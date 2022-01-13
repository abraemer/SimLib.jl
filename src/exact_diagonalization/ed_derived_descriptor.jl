"""
    abstract EDDerivedDataDescriptor <: SimLib.AbstractDataDescriptor

Supertype for descriptors of quantities that derive from exact diagonalization.

## Interface
 - Subtypes must have a field `derivedfrom` holding the `EDDataDescriptor` they derived from
 - Subtypes must implement [`_default_folder`](@ref) or [`SimLib._filename`](@ref)
 - Subtypes may implement [`_filename_addition`](@ref)

This automatically forwards fields not found in the subtype to the `EDDataDescriptor`.
"""
abstract type EDDerivedDataDescriptor <: SimLib.AbstractDataDescriptor end

# simply forward all properties, that are directly part of the descriptor
function Base.getproperty(eddd::EDDerivedDataDescriptor, p::Symbol)
    if hasfield(typeof(eddd), p)
        return getfield(eddd, p)
    else
        return getproperty(getfield(eddd, :derivedfrom), p)
    end
end

"""
    _filename(::EDDerivedDataDescriptor)

Filename consists of a default folder for the quantity followed by a filename given by the model's parameters
and possible some addition from the quantity.
"""
function SimLib._filename(eddd::EDDerivedDataDescriptor)
    return joinpath(
        _default_folder(eddd),
        model_fileprefix(eddd.model) * _filename_addition(eddd))
end

"""
    _default_folder

Return default folder to store results for the quantity. E.g. "lsr" for level-spacing ratio.
"""
function _default_folder end
_filename_addition(::EDDerivedDataDescriptor) = ""
