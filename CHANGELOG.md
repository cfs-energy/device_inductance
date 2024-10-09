# Changelog

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
