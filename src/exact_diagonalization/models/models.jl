module EDModels

using SparseArrays
#using ...Positions
using ...SimLib
using SpinSymmetry
using Printf: @sprintf
using XXZNumerics

export parameter_dims, initialize_model!, split_workload, do_parameters, model_name, model_fileprefix
export RandomPositionsXXZWithXField, RandomPositionsXXZWithDegeneracyLifted

abstract type Model end

"""
    parameter_dims(model) -> (dim1, dim2, ...)
"""
function parameter_dims end


"""
    initialize_model!(model, array_initializer)
"""
function initialize_model! end


"""
    split_workload(model, num_parts) -> [parameter_chunk...]

`parameter_chunk` is put through `do_parameters` to iterate all Hamiltonians in this section of the parameter space.
"""
function split_workload end#(model, num_parts(runmode))


"""
    do_parameters(f, model, parameter_chunk)

f = f(parameter_index, H)

Construct all Hamiltonians associated with the parameter_chunk and call `f` with the index and the Matrix.
"""
function do_parameters end

"""
    model_name(model)

Return a nice name that could be used as foldername :)
"""
function model_name end

"""
    model_fileprefix(model)

Return a nice name that could prefix a file with data derived from this model.
"""
function model_fileprefix end

include("Jxxz_x.jl")
include("Jxxz_Jz.jl")
#include("xxz_hx.jl")

end #module
