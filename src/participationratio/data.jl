### Descriptor

struct PRDataDescriptor{B} <: ED.EDDerivedDataDescriptor
    basis::B
    derivedfrom::ED.EDDataDescriptor
end



PRDataDescriptor(basis, args...; kwargs...) = PRDataDescriptor(basis, EDDataDescriptor(args...; kwargs...))



### Data obj

struct PRData{B,N} <: SimLib.AbstractSimpleData
    descriptor::PRDataDescriptor{B}
    data::FArray{N}
end

ED._default_folder(::PRDataDescriptor) = "pr"
ED._filename_addition(prdd::PRDataDescriptor) = "_"*referencebasis_name(prdd.basis)

"""
    load_pr(basis, edd)
    load_pr(basis, model[, diagtype][, location])
"""
load_pr(args...; kwargs...) = load(PRDataDescriptor(args...; kwargs...))
