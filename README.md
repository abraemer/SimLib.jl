# SimLib

This package contains all the ingredients to run simulations of disordered spin systems. 

## Basic concept
Each step of the simulation (with the exception of `ED`) has two types associated with it: A `Descriptor` holding all relevant parameters and a `Data` type which contains the descriptor and the computed values. Usually the workflows look like this: You initialize a `Descriptor` with the parameters of interest and either `create` or `load` the corresponding `Data` (depending  on whether you ran the simulation already or not). When `creat`ing new results, the code will try to `load` required data for the computation (f.e. the ensemble computation need eigenstate occupation, operator diagonal and eigenenergy data) if you did not pass in data explicitely.

Note: It's not really possible to pass additional data to `create`. Right now there are (sometimes) specific methods (i.e. `ensemble_predictions`) for the given computation, but `create` never accepts additional arguments. This should probably be changed to unify the interface of the computational tasks.

## Exact diagonalization
There is an exception with computations that require Exact Diagonalization. It is a lot more efficient to bundle computations s.t. you only perform the diagonalization once and reuse it for different quantities of interest. To facilitate this, you can specify the parameters once (using `EDDataDescriptor`) and then call `run_ed` with a list of `Task`s to fulfill. This results in a list of `Data` structs - one for each task.

## Saving/Loading
Every descriptor has a `SaveLocation` to control where the data should be saved to/loaded from and knows how to generate a good file from it's values. The `SaveLocation(prefix, suffix)` denotes the path `prefix/<descriptor-specific-name>_suffix.jld2`. Derived (or otherwise related) data also keep that information. 

All methods dealing with file operations take also `prefix`, `suffix` keywords which take precendence over the `SaveLocation` of the `Descriptor` at hand.

## Tests
There are also some very basic tests around, running all simulation steps for some small parameters, and some notebooks to visualize raw data (useful for finding bugs).

## Example
For a small example showcasing a simulation script, see `notebooks/generate_test_data.jl` (which actually not a `Pluto.jl` notebook and probably should be in a different place).

## Implemented quantities
 - Energies (= levels)
 - eigenstate occupation
 - operator diagonal (= eigenstate expectation values)
 - Half-chain entropy (arb. subsystem lengths, only for zblock basis)
 - eigenstate locality
 - inverse participation ratio (basic)
 - Level spacing ratio
 - Ensemble prediction

Every quantity (except the last 2) need to be computed as an EDTask! LSR can be computed via a Task.