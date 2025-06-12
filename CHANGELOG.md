# Changelog

## 2.1.0 - 2025-06-12

### Changed

* Migrate to uv from poetry

## 2.0.0 - 2025-05-02

Overhaul structure discretization strategy and improve testing of structures and structure model reduction.
* Add multi-level discretization to reduce number of distinct structure elements
* Improve testing of structure discretization to provide sub-1% match on important parameters

### Added

* Add `structures` subpackage & split major components of structure discretization into separate files
* Add multi-level structure discretization
  * Inputs: combined cross-sectional representations of wall and pf_passive components
    * Wall "elements" in each wall "section" treated the same as a pf_passive "element"
  * Loops: One or more coarse chunk(s) of input cross-sections
    * Chunked radially about the limiter centroid to prioritize preservation of structure-plasma interaction
    * Inputs are only chunked into multiple loops if the input structure takes up a large angular span relative to the limiter centroid AND has a large perimeter-to-area ratio; otherwise, the input is treated as a single loop.
      * This results in detecting and chunking the vacuum vessel, but not smaller or blockier structures like coil supports
  * Filaments: Thin-filament representation of the result of meshing each loop
    * Each loop owns many filaments, and its aggregate inductances, resistance, flux and B-field, etc. are calculated using those filaments as the source points
* Add tests of structure discretization invariants
  * Total system stored energy per unit cross-sectional current density and total parallel resistance checked for invariance under changing discretization coarseness
  * Structure model reduction eigenvalues checked for consistency - some small change is expected here, but not much

### Changed

* !`DeviceInductance.structures` now returns a `list[PassiveStructureLoop]` instead of `list[PassiveStructureFilament]`
* !`DeviceInductance` init now requires keyword arguments for all optional arguments (everything except the always-required ODS device description)
* !Remove `DeviceInductance.structure_filament_rz()` function which no longer refers to valid fields
* Set readme link in pyproject.toml
* Update readme image to explanatory poster
* Round eigenvalues to 16 decimal places for nonnegativity check in `stabilized_eigenmode_reduction()`
  * Some eigenvalues can come out slightly negative (around -1e-20) due to numerical error
* Update default device to include example full flux loop sensor
  * Update sensor tests to assert presence of all sensor types
* Increment coverage fail-under to 96%
* Check more structure inductances with grad-shafranov method & remove the extremely slow and redundant filamentized self-inductance check
* Remove stale test-only functions
* Tighten tolerances on model reduction dI/dt check to 0.1% for PF/CS and 1% for DV/VS
  * Use all modes for testing to avoid consuming truncation error, which is up to the user - we're only testing whether the approach to model transformation is correct, not whether a given level of truncation is acceptable for a given application

## 1.9.1 - 2025-03-27

### Fixed

* Fix bug in assembly of circuit mutual inductance matrix which caused only half the inductance terms of each circuit to be accounted

### Added

* Add more rigorous test of circuit mutual inductance matrix using flux interpolator method

## 1.9.0 - 2025-03-19

### Added

* Add force matrices describing force applied to coils by each
  coil, circuit, conducting structure filament, conducting structure mode, and plasma mesh cell
  * Two options for plasma-coil forces, one that requires the full plasma flux tables and produces
    a matrix covering the full domain, and another that does a direct calculation from only the points
    inside the limiter.
* Add logging
* Add support for python 3.13

### Changed

* Update gmsh settings for passive structure meshing to be more deterministic
  * Reduce randomization factor, fix random seed, and use single thread so that RNGs do not get out of sync between threads
* Sort structure filaments by descending L/R timescale to improve conditioning of model reduction

## 1.8.0 - 2025-02-12

### Added

* `utils.flux_solver` and `utils.solve_flux_axisymmetric` functions extracted from device for solving continuous flux fields by finite difference
* `coils.Coil.[grids, meshes, extent, local_flux_table]` cached properties for solving a smooth self-field over coil winding packs if they can be mapped on to a regular grid

### Changed

* Plasma flux functions defer to new flux solver functions in `utils`
* Update coil flux table procedure to patch coil local field from coil-specific Grad-Shafranov solve over region near winding pack
* Add sensible defaults for mesh extent and resolution (coil extent + 0.1m pad and 0.05m resolution)

## 1.7.4 - 2024-10-09

### Changed

* Update readthedocs config

## 1.7.3 - 2024-10-08

### Changed

* Add table of methods to docs

## 1.7.2 - 2024-10-08

### Changed

* Improve docstrings

## 1.7.1 - 2024-10-03

### Changed

* Update public device description to the exact released version

## 1.7.0 - 2024-09-30

### Added

* Add mkdocs API documentation
* Add load_default_ods function to load example description
* Add open-source version of device description to examples

### Changed

* Docstring, formatting, and type annotation updates
* Send some backend functions formally to private API
* Update examples and tests for new example description

## 1.6.0 - 2024-09-25

### Added

* Add `contour` module with specialized boundary-tracing function 
* Add calculation of plasma self-inductance
* Extract 4th-order gradient method to its own function in the utils module

### Changed

* Update sensors to calculate their response as the ideal integrated response instead of an instantaneous voltage reading
* Smoketest examples automatically via pytest instead of manually listing in workflow
* Slightly tighten coverage fail-under to 94%

## 1.5.0 - 2024-08-19

### Added

* Add `sensors` module with full and partial flux loops and bpol coils
* Implement voltage response functions for magnetic sensors

## 1.4.2 - 2024-08-02

### Changed

* Update deps and readme
* Add support for python 3.12
* Resolve numpy deprecation warnings

## 1.4.1 - 2024-06-20

### Changed

* Update CI and job runner

## 1.4.0 - 2024-04-24

### Changed

* Add circuit names

## 1.3.1 - 2024-04-16

### Fixed

* Fix `DeviceInductance.__post_init__` not being called now that it is no longer a dataclass

## 1.3.0 - 2024-04-10

### Changed

* Simplify DeviceInductance class using cached properties
  * No change to public API
  * Reduced line count & complexity
  * Makes DeviceInductance formally immutable (except interior mutation) after init, to prevent errors
* Use cached properties to back .get_coil_names() and .get_coil_index_dict() so that they don't hamstring loops

### Added

* Add .coil_names and .coil_index_dict properties to DeviceInductance for convenience

## 1.2.0 - 2024-04-09

### Changed

* Use diverging colormaps in examples/full_workflow
* Freeze coil and structure dataclasses to prevent accidental mutation

### Added

* Add `circuits` module for parsing circuit info from ODS input
* Add `DeviceInductance.circuits` and related fields for mutual inductances, resistances, flux tables, and B-field tables
* Add `DeviceInductance.n_coils,.n_structures,.n_circuits,.nr,.nz` properties for convenience
* Add `DeviceInductance.get_coil_names,.get_coil_index_map` methods for convenience
* Add CHANGELOG.md

## 1.1.1 - 2024-04-04

### Fixed

* Fixed poetry dependency format in pyproject.toml to capture internal deps during pypi deployment

### Changed

* Updated readme to include libglu1 dep

## 1.1.0 - 2024-03-29

### Fixed

* Fix stabilized eigenmode model reduction method to be proper PCA and do scalings in the right order
  * Should take sqrt of eigenvalues, not elementwise on the original matrix
  * Row scaling T has no analytic basis and reduces quality-of-fit

### Changed

* Update interface between "typical workflow" outputs and device to make
  sure that the final extent of the computational domain is always the one
  that ends up in the TypicalOutputs
* Improve example plots
* Rename DeviceInductance.mode_resistances to .structure_mode_resistances for consistency
  * Technically a breaking API change, but I don't think anyone's using the API yet

### Added

* Add more transformed matrices as properties
  * Mode-mode and mode-coil mutual inductances
  * Helps offload the user's need to understand how to apply the model
    reduction transform
* Add B-field tables
  * Circular-filament calcs with a 15cm radius patch around the current
    source replaced with the result of a 4th order finite difference calc on
    the flux grid to remove numerical wonkiness of filament calc near the
    singularity
  * The flux field is also made using a filament calc, but it's
    numerically better-conditioned due to factors of 1/r cancelling out with
    the area integral
* Add plasma flux and B-field calc methods
  * Both summation and G-S solve; operator factorization cached to reduce
    compute overhead for repeated solves
* Add calc_flux_density_from_flux
  * Improved version of calc_B_from_psi method from MARS
  * Tested against filament calcs during B-field table tests
* Add limiter and limiter mask properties
* Did not add force tables
  * Can only tabulate force w.r.t. a filament in a certain orientation
  * Takes a lot of context to use the tables correctly and very little to
    build them
  * Highly error-prone and doesn't save us much time (5 lines and 3ms of
    processing time)
  * Might add functions for calculating force-per-amp on a specific set of
    input filaments at some point instead
* Add extent_for_plotting property
