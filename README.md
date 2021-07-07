# SimLib

This package contains the scripts to actually run the simulations. It does not contain test as the scripts using the functionality are test enough... or so I hope. 

## Structure

- `src/` contains different components that may be reused. Basically the core of every step of the simulation is written as a small module providing the possibility to save/load/compute the dataset it's responsible for.
- Files in `slurm/` maybe run directly (or given to `sbatch`) to perform an action.