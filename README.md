# SimLib

This package contains the scripts to actually run the simulations. It does not contain some tests that just try to run the most basic functionality with some parameters (chosen for short runtime).

## File Structure

- `src/` contains different components that may be reused. Basically the core of every step of the simulation is written as a small module providing the possibility to save/load/compute the dataset it's responsible for.
- Files in `slurm/` maybe run directly (or given to `sbatch`) to perform an action.
- `test/` has the basic test scripts.

## Code
The main idea is, that each type of data that interests me has its own `Data` type. These types consist of some arrays to store the values and a `Descriptor` that holds the parameters for obtaining the data (and then some more helpful stuff). The `Descriptor` is responsible to know where to `save` to/`load` from and how to `create` the `Data` in the first place. This design let's one to specify the data one wants with all necessary parameters and then just call `create` it. Intermediate `Data` used for computation will be automatically `load_or_created` and also saved. Loading data is simplified as not all parameters are important to know the save location.