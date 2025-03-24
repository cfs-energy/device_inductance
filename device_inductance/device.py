from __future__ import annotations

from typing import Literal, Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from cfsem import (
    self_inductance_distributed_axisymmetric_conductor,
)

from omas import ODS

from shapely import Polygon, Point

from device_inductance.structures import PassiveStructureFilament, _extract_structures
from device_inductance.coils import Coil, _extract_coils
from device_inductance.circuits import CoilSeriesCircuit, _extract_circuits
from device_inductance.sensors import (
    PoloidalFieldProbe,
    FullFluxLoop,
    PartialFluxLoop,
    _extract_full_flux_loops,
    _extract_partial_flux_loops,
    _extract_poloidal_field_probes,
)
from device_inductance.mutuals import (
    _calc_coil_mutual_inductances,
    _calc_structure_mutual_inductances,
    _calc_coil_structure_mutual_inductances,
    _calc_circuit_mutual_inductances,
    _calc_circuit_structure_mutual_inductances,
)
from device_inductance.tables import (
    _calc_coil_flux_tables,
    _calc_coil_flux_density_tables,
    _calc_structure_flux_tables,
    _calc_structure_flux_density_tables,
    _calc_structure_mode_flux_tables,
    _calc_structure_mode_flux_density_tables,
    _calc_mesh_flux_tables,
    _calc_circuit_flux_tables,
    _calc_circuit_flux_density_tables,
)
from device_inductance.forces import (
    _calc_coil_coil_forces,
    _calc_circuit_coil_forces,
    _calc_structure_coil_forces,
    _calc_plasma_coil_forces,
)
from device_inductance.utils import (
    calc_flux_density_from_flux,
    flux_solver,
    solve_flux_axisymmetric,
    _join_extents,
    _pad_extent,
)
from device_inductance import model_reduction
from device_inductance.logging import log, logger_is_set_up, logger_setup_default

F64 = np.float64


class DeviceInductance:
    """
    Thin wrapper to provide methods and properties on a device description ODS.
    Extracts geometries and calculates inductance matrices as well as flux fields
    and B-fields on the generated regular grid.

    Note:
        The contents of an instance of this class must be treated as
        immutable, as it relies on caching results to eliminate repeated
        computations along different paths through the analysis. Do not
        modify the supplied ODS after initialization or modify any of the
        values returned from methods or properties of this class!
    """

    _ods: ODS
    """OMAS data object in the format produced by device_description"""
    _max_nmodes: int = 40
    """Maximum number of structure modes to retain"""
    _min_extent: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """
    [m] rmin, rmax, zmin, zmax extent of computational domain.
    This will be updated during mesh initialization, during which it
    may be adjusted to satisfy the required spatial resolution.
    """
    _dxgrid: tuple[float, float] = (0.05, 0.05)
    """[m] spatial resolution of computational grid"""
    _model_reduction_method: Literal["eigenmode", "stabilized eigenmode"] = "eigenmode"
    """Choice of method for truncating passive structure system modes"""
    _show_prog: bool = False
    """Whether to display terminal progress bars during expensive calculations"""
    _plasma_coil_force_method: Literal["tables", "mask"] = "mask"
    """Whether to interpolate B-field on the fully-realized mesh tables,
    or do direct filament calculations from points inside the limiter mask.
    Defaults to "mask", which is faster and uses less memory, but only includes
    nonzero entries inside the limiter, which requires a valid limiter geometry."""

    def __init__(
        self,
        ods: ODS,
        max_nmodes: int = 40,
        min_extent: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        dxgrid: tuple[float, float] = (0.05, 0.05),
        model_reduction_method: Literal[
            "eigenmode", "stabilized eigenmode"
        ] = "eigenmode",
        show_prog: bool = False,
        plasma_coil_force_method: Literal["tables", "mask"] = "mask",
        **kwargs,  # For backwards compatibility with `extent` kwarg only
    ):
        """
        Attach a new DeviceInductance instance to an existing ODS description.

        Args:
            ods: OMAS data object in the format produced by device_description
            max_nmodes: Maximum number of structure modes to retain
            min_extent: [m] rmin, rmax, zmin, zmax extent of computational domain.
                    This will be updated during mesh initialization, during which it
                    may be adjusted to satisfy the required spatial resolution.
            dxgrid: [m] spatial resolution of computational grid
            show_prog: Whether to display terminal progress bars during expensive calculations
            plasma_coil_force_method: Whether to interpolate B-field on the fully-realized mesh tables,
                                      or do direct filament calculations from points inside the limiter mask.
                                      Defaults to "mask", which is faster and uses less memory, but only includes
                                      nonzero entries inside the limiter, which requires a valid limiter geometry.
        """
        if not logger_is_set_up():
            logger_setup_default()

        if "extent" in kwargs.keys():
            # Backwards compatibility with `extent` kwarg name only
            min_extent = kwargs.pop("extent")

        if len(kwargs) != 0:
            log().warning(f"DeviceInductance init ignoring extra kwargs: {kwargs}")

        self._ods = ods
        self._max_nmodes = max_nmodes
        self._min_extent = min_extent
        self._dxgrid = dxgrid
        self._model_reduction_method = model_reduction_method
        self._show_prog = show_prog
        self._plasma_coil_force_method = plasma_coil_force_method

        self.__post_init__()

    def __hash__(self) -> int:
        # Do not need to hash immutable inputs
        return hash(id(self))

    def __post_init__(self):
        # Set a sensible default that encompasses the coils with 0.1m pad if none was provided
        if self._min_extent == (0.0, 0.0, 0.0, 0.0):
            inf = float("inf")
            self._min_extent = (inf, -inf, inf, -inf)
            for c in self.coils:
                self._min_extent = _join_extents(self._min_extent, c.extent)
            self._min_extent = _pad_extent(self._min_extent, (0.1, 0.1))
            # Don't auto-place grid cells too close to R=0
            rmin, rmax, zmin, zmax = self._min_extent
            rmin = max(rmin, 0.1)
            self._min_extent = (rmin, rmax, zmin, zmax)

        # Set a sensible default grid resolution to avoid dividing by zero
        if self._dxgrid == (0.0, 0.0):
            self._dxgrid = (0.05, 0.05)

        # Immutable after init, except for new cache entries
        def setattr_err(*_, **__):
            raise NotImplementedError(
                "DeviceInductance attributes are not intended to be mutated"
            )

        self.__setattr__ = setattr_err

    @property
    def ods(self) -> ODS:
        """
        The ODS description object with the device description.
        This object must not be edited after the DeviceInductance object
        is initialized!
        """
        return self._ods

    @property
    def max_nmodes(self) -> int:
        """Maximum number of (eigen)modes to keep in the passive structure model reduction"""
        return self._max_nmodes

    @property
    def min_extent(self) -> tuple[float, float, float, float]:
        """[m] (rmin, rmax, zmin, zmax) minimum extent requested at init.
        The actual extent of the mesh bounds this minimum, while respecting the
        required grid resolution exactly.
        """
        return self._min_extent

    @property
    def dxgrid(self) -> tuple[float, float]:
        """[m] (dr, dz) grid cell spacing"""
        return self._dxgrid

    @property
    def model_reduction_method(self) -> Literal["eigenmode", "stabilized eigenmode"]:
        """Choice of passive structure state-space model dimensionality reduction method"""
        return self._model_reduction_method

    @property
    def show_prog(self) -> bool:
        """Whether terminal progressbar will be displayed during some calcs"""
        return self._show_prog

    @cached_property
    def coils(self) -> list[Coil]:
        """Info about coil name and filaments"""
        return _extract_coils(self.ods)

    @cached_property
    def structures(self) -> list[PassiveStructureFilament]:
        """Info about structure filaments"""
        return _extract_structures(self.ods, self.show_prog)

    @cached_property
    def circuits(self) -> list[CoilSeriesCircuit]:
        """Info about coils wired in series"""
        return _extract_circuits(self.ods)

    @cached_property
    def limiter(self) -> Polygon:
        """[m] Plasma-facing first-wall contour defined as a closed path around the plasma"""
        # This one is pretty quick, doesn't need its own file
        limiter_path_r = self.ods["wall.description_2d.0.limiter.unit.0.outline.r"]
        limiter_path_z = self.ods["wall.description_2d.0.limiter.unit.0.outline.z"]
        return Polygon(zip(limiter_path_r, limiter_path_z))

    @cached_property
    def poloidal_field_probes(self) -> list[PoloidalFieldProbe]:
        """List of poloidal B-field probe sensors"""
        return _extract_poloidal_field_probes(self.ods)

    @cached_property
    def full_flux_loops(self) -> list[FullFluxLoop]:
        """List of full flux loop sensors"""
        return _extract_full_flux_loops(self.ods)

    @cached_property
    def partial_flux_loops(self) -> list[PartialFluxLoop]:
        """List of partial flux loop sensors"""
        return _extract_partial_flux_loops(self.ods)

    @cached_property
    def limiter_mask(self) -> NDArray[F64]:
        """Mask with shape (nr, nz) indicating which parts of the mesh are inside the limiter"""
        limiter = self.limiter
        rmesh, zmesh = self.meshes
        nr, nz = rmesh.shape
        mask = np.ones_like(rmesh)
        # This is slow, but shapely only accepts one
        # point at a time
        for i in range(nr):
            for j in range(nz):
                r = rmesh[i, j]
                z = zmesh[i, j]
                p = Point(r, z)
                mask[i, j] *= limiter.contains(p)
        return mask

    @cached_property
    def _calc_meshes(
        self,
    ) -> tuple[tuple[NDArray[F64], NDArray[F64]], tuple[float, float, float, float]]:
        """Initialize both meshes and final extent after adjustment to achieve target resolution"""
        # Actualize the grid/mesh and update extent
        rmin, rmax, zmin, zmax = self.min_extent  # [m]
        dr, dz = self.dxgrid  # [m]
        if rmax - rmin > 0.0 and zmax - zmin > 0.0:
            rgrid = np.arange(rmin, rmax + dr, dr)  # [m] Grid that spans full extent
            zgrid = np.arange(zmin, zmax + dz, dz)  # [m]
            rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")  # [m]

            # Update the extent, which may have been adjusted
            extent = (float(np.min(rgrid)), float(np.max(rgrid)), float(np.min(zgrid)), float(np.max(zgrid)))
        else:
            # If our grid spec is zero-size, make a unit mesh
            # to allow the calcs to proceed, without providing real tables
            rmesh = np.nan * np.zeros((1, 1))
            zmesh = np.nan * np.zeros((1, 1))
            extent = self.min_extent
        return (rmesh, zmesh), extent

    @cached_property
    def meshes(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[m] Shape: ((nr, nz), (nr, nz)) 2D meshgrids over domain"""
        meshes, _ = self._calc_meshes
        return meshes

    @cached_property
    def grids(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[m] with shape ((nr), (nz)) 1D grids"""
        meshes, _ = self._calc_meshes
        rgrid = meshes[0][:, 0]  # [m] all rs for first z
        zgrid = meshes[1][0, :]  # [m] all zs for first r
        return (rgrid, zgrid)

    @cached_property
    def extent(self) -> tuple[float, float, float, float]:
        """[m] The (rmin, rmax, zmin, zmax) extent of the grid cell centers"""
        _, extent = self._calc_meshes
        return extent

    @cached_property
    def extent_for_plotting(self) -> tuple[float, float, float, float]:
        """[m] The (rmin, rmax, zmin, zmax) extent of the grid cell edges"""
        rmin, rmax, zmin, zmax = self.extent
        dr, dz = self.dxgrid
        return (
            rmin - dr / 2,
            rmax + dr / 2,
            zmin - dz / 2,
            zmax + dz / 2,
        )

    @cached_property
    def nr(self) -> int:
        """Number of r-discretizations in mesh"""
        return self.grids[0].size

    @cached_property
    def nz(self) -> int:
        """Number of z-discretizations in mesh"""
        return self.grids[1].size

    @cached_property
    def n_coils(self) -> int:
        """Number of coils"""
        return len(self.coils)

    @cached_property
    def n_structures(self) -> int:
        """Number of structure filaments"""
        return len(self.structures)

    @cached_property
    def n_structure_modes(self) -> int:
        """Number of structure modes retained"""
        return len(self.structure_mode_eigenvalues)

    @cached_property
    def n_circuits(self) -> int:
        """Number of coil series circuits"""
        return len(self.circuits)

    @cached_property
    def coil_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (ncoil X ncoil), Coil-coil mutual inductance"""
        return _calc_coil_mutual_inductances(self.coils, self.show_prog)

    @cached_property
    def structure_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (nstruct X nstruct), Structure-structure mutual inductance"""
        return _calc_structure_mutual_inductances(self.structures, self.show_prog)

    @cached_property
    def structure_mode_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (nmodes X nmodes), Structure mode-mode mutual inductance"""
        mss = self.structure_mutual_inductances
        tuv = self.structure_model_reduction
        return tuv.T @ (mss @ tuv)

    @cached_property
    def coil_structure_mode_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (ncoil X nmodes), Coil-to-structure-mode mutual inductance"""
        mcs = self.coil_structure_mutual_inductances
        tuv = self.structure_model_reduction
        return mcs @ tuv

    @cached_property
    def coil_structure_mutual_inductances(self) -> NDArray[F64]:
        """[H] (ncoil X nstruct) Coil-structure mutual inductance"""
        return _calc_coil_structure_mutual_inductances(
            self.coils, self.structures, self.show_prog
        )

    @cached_property
    def circuit_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (ncirc X ncirc), Circuit-circuit mutual inductance"""
        return _calc_circuit_mutual_inductances(
            self.circuits, self.coil_mutual_inductances, self.show_prog
        )

    @cached_property
    def circuit_structure_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (ncirc X nstruct), Circuit-structure mutual inductance"""
        return _calc_circuit_structure_mutual_inductances(
            self.circuits, self.coil_structure_mutual_inductances, self.show_prog
        )

    @cached_property
    def circuit_structure_mode_mutual_inductances(self) -> NDArray[F64]:
        """[H] with shape (ncirc X nmodes), Circuit-to-structure-mode mutual inductance"""
        mcircs = self.circuit_structure_mutual_inductances
        tuv = self.structure_model_reduction
        return mcircs @ tuv

    @cached_property
    def coil_resistances(self) -> NDArray[F64]:
        """[ohm] with shape (ncoil X ncoil), Resistance of coils as diagonal matrix"""
        return np.diag(np.array([c.resistance for c in self.coils]))

    @cached_property
    def structure_resistances(self) -> NDArray[F64]:
        """[ohm] (nstruct X nstruct) Resistance of structure fils, diagonal"""
        return np.diag(np.array([s.resistance for s in self.structures]))

    @cached_property
    def structure_mode_resistances(self) -> NDArray[F64]:
        """[H] with shape (ncoil X nstruct), Coil-structure mutual inductance"""
        r = self.structure_resistances
        tuv = self.structure_model_reduction
        return tuv.T @ (r @ tuv)

    @cached_property
    def circuit_resistances(self) -> NDArray[F64]:
        """[ohm] with shape (ncirc X ncirc), Resistance of circuits as diagonal matrix"""
        r = np.zeros(self.n_circuits)
        # Sum the series resistance of the coils in the circuit
        for i in range(self.n_circuits):
            r[i] = np.sum(
                [self.coil_resistances[c[0], c[0]] for c in self.circuits[i].coils]
            )
        return np.diag(r)

    @cached_property
    def structure_mode_eigenvalues(self) -> NDArray[F64]:
        """
        [s] with shape (neig), eigenvalues of structure model reduction,
        kept for comparison purposes only, as these are not
        needed in order to use the structure model reduction.
        """
        d, _, _ = self._calc_model_reduction
        return d

    @cached_property
    def structure_model_reduction(self) -> NDArray[F64]:
        """
        [dimensionless] with shape (nstruct X neig), structure model-reduction matrix.\n
        Each column is a normalized vector of unit length.

        Apply this transform like `a_transformed = a @ tuv` for matrices
        with `a.shape == (..., nstruct)`.\n
        In the case of diagonalized structure filament resistances,\n
        the transform is applied like `r_transformed = tuv.T @ (r @ tuv)`.
        """
        _, tuv, _ = self._calc_model_reduction
        return tuv

    @cached_property
    def coil_flux_tables(self) -> NDArray[F64]:
        """[Wb/A] with shape (ncoils X nr X nz), Coil flux tables"""
        return _calc_coil_flux_tables(self.coils, self.meshes, self.show_prog)

    @cached_property
    def structure_flux_tables(self) -> NDArray[F64]:
        """[Wb/A] with shape (nstruct X nr X nz), Structure flux tables"""
        return _calc_structure_flux_tables(self.structures, self.meshes, self.show_prog)

    @cached_property
    def structure_mode_flux_tables(self) -> NDArray[F64]:
        """[Wb/A] with shape (nmodes X nr X nz), Structure mode flux tables"""
        return _calc_structure_mode_flux_tables(
            self.structure_flux_tables,
            self.structure_model_reduction,
            self.show_prog,
        )

    @cached_property
    def plasma_flux_tables(self) -> NDArray[F64]:
        """[Wb/A] with shape (nr*nz X nr X nz), Plasma flux tables"""
        return _calc_mesh_flux_tables(self.meshes, self.grids, self.show_prog)

    @cached_property
    def circuit_flux_tables(self) -> NDArray[F64]:
        """[Wb/A] with shape (ncirc X nr X nz), Circuit flux tables"""
        return _calc_circuit_flux_tables(
            self.circuits, self.coil_flux_tables, self.show_prog
        )

    @cached_property
    def coil_flux_density_tables(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[T/A] with shape (ncoils X nr X nz), Coil flux density (B-field) tables, r- and z- components"""
        return _calc_coil_flux_density_tables(
            self.coils, self.meshes, self.coil_flux_tables, self.show_prog
        )

    @cached_property
    def structure_flux_density_tables(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[T/A] with shape (nstruct X nr X nz), Structure flux density (B-field) tables, r- and z- components"""
        return _calc_structure_flux_density_tables(
            self.structures, self.meshes, self.structure_flux_tables, self.show_prog
        )

    @cached_property
    def structure_mode_flux_density_tables(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[T/A] with shape (nstruct X nr X nz), Structure mode flux density (B-field) tables, r- and z- components"""
        return _calc_structure_mode_flux_density_tables(
            *self.structure_flux_density_tables,
            self.structure_model_reduction,
            self.show_prog,
        )

    @cached_property
    def circuit_flux_density_tables(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """[T/A] with shape (ncirc X nr X nz), Circuit flux density (B-field) tables, r- and z- components"""
        return _calc_circuit_flux_density_tables(
            self.circuits,
            *self.coil_flux_density_tables,
            self.show_prog,
        )

    @cached_property
    def coil_coil_forces(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        [N/A^2] with shape (ncoils X ncoils), Coil-coil force tables, r- and z- components.
        
        Note that analytic force estimates include a variety of sources of error, including
        geometric differences between analysis and real hardware, discretization error,
        numerical summation error, and so on. Due to the importance of loads analysis, it
        is always recommended to double-check force results with at least one other tool,
        preferably of meaningfully distinct design - for example, cross-check an analytic method
        with a finite-element method.

        Because the accuracy of force estimates depends heavily on problem setup and geometry,
        no particular claims are made here about the accuracy of the force calculations,
        and they should never be used for human safety applications.
        """
        return _calc_coil_coil_forces(
            self.coils, self.grids, self.coil_flux_density_tables, self.show_prog
        )

    @cached_property
    def circuit_coil_forces(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        [N/A^2] with shape (ncirc X ncoils), Circuit-coil force tables, r- and z- components.
        
        Note that analytic force estimates include a variety of sources of error, including
        geometric differences between analysis and real hardware, discretization error,
        numerical summation error, and so on. Due to the importance of loads analysis, it
        is always recommended to double-check force results with at least one other tool,
        preferably of meaningfully distinct design - for example, cross-check an analytic method
        with a finite-element method.

        Because the accuracy of force estimates depends heavily on problem setup and geometry,
        no particular claims are made here about the accuracy of the force calculations,
        and they should never be used for human safety applications.
        """
        return _calc_circuit_coil_forces(
            self.coils, self.circuits, self.coil_coil_forces, self.show_prog
        )

    @cached_property
    def structure_coil_forces(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        [N/A^2] with shape (nstruct X ncoils), Structure filament-coil force tables, r- and z- components.
        
        Note that analytic force estimates include a variety of sources of error, including
        geometric differences between analysis and real hardware, discretization error,
        numerical summation error, and so on. Due to the importance of loads analysis, it
        is always recommended to double-check force results with at least one other tool,
        preferably of meaningfully distinct design - for example, cross-check an analytic method
        with a finite-element method.

        Because the accuracy of force estimates depends heavily on problem setup and geometry,
        no particular claims are made here about the accuracy of the force calculations,
        and they should never be used for human safety applications.
        """
        return _calc_structure_coil_forces(self.coils, self.structures, self.show_prog)

    @cached_property
    def structure_mode_coil_forces(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        [N/A^2] with shape (nmodes X ncoils), Structure mode-coil force tables, r- and z- components.
        
        Note that analytic force estimates include a variety of sources of error, including
        geometric differences between analysis and real hardware, discretization error,
        numerical summation error, and so on. Due to the importance of loads analysis, it
        is always recommended to double-check force results with at least one other tool,
        preferably of meaningfully distinct design - for example, cross-check an analytic method
        with a finite-element method.

        Because the accuracy of force estimates depends heavily on problem setup and geometry,
        no particular claims are made here about the accuracy of the force calculations,
        and they should never be used for human safety applications.
        """
        tuv = self.structure_model_reduction
        frsc, fzsc = self.structure_coil_forces
        fr = tuv.T @ frsc
        fz = tuv.T @ fzsc
        return (fr, fz)

    @cached_property
    def plasma_coil_forces(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        [N/A^2] with shape (nr*nz X ncoils), Mesh-coil force tables, r- and z- components.
        
        Note that analytic force estimates include a variety of sources of error, including
        geometric differences between analysis and real hardware, discretization error,
        numerical summation error, and so on. Due to the importance of loads analysis, it
        is always recommended to double-check force results with at least one other tool,
        preferably of meaningfully distinct design - for example, cross-check an analytic method
        with a finite-element method.

        Because the accuracy of force estimates depends heavily on problem setup and geometry,
        no particular claims are made here about the accuracy of the force calculations,
        and they should never be used for human safety applications.
        """
        if self._plasma_coil_force_method == "tables":
            plasma_flux_tables_or_limiter_mask = self.plasma_flux_tables
            full_flux_tables = True
        elif self._plasma_coil_force_method == "mask":
            plasma_flux_tables_or_limiter_mask = self.limiter_mask
            full_flux_tables = False
        else:
            raise ValueError("Unexpected calc method")

        return _calc_plasma_coil_forces(
            self.coils,
            self.grids,
            self.meshes,
            plasma_flux_tables_or_limiter_mask,
            full_flux_tables,
            show_prog=self.show_prog,
        )

    @cached_property
    def structure_filament_rz(self) -> list[tuple[float, float]]:
        """
        [m] r,z coordinates of structure filaments.
        Provided to prevent repeatedly assembling this representation when evaluating plasma inductances.
        """
        rz = [(x.r, x.z) for x in self.structures]
        return rz  # [m]

    @cached_property
    def coil_filament_rzn(self) -> list[list[tuple[float, float, float]]]:
        """
        ([m], [m], [dimensionless]) r,z,n of each coil's filaments, where `n` is number of turns.
        Provided to prevent repeatedly assembling this representation when evaluating plasma inductances.
        """
        rzn = []
        for coil in self.coils:
            rzn.append([(x.r, x.z, x.n) for x in coil.filaments])
        return rzn  # [m], [m], [dimensionless]

    @cached_property
    def flux_solver(self) -> Callable[[NDArray[F64]], NDArray[F64]]:
        """
        Linear solver for extracting a flux field from a toroidal current density distribution
        using a 4th-order finite difference approximation of the Grad-Shafranov PDE.
        For `jtor` toroidal current density shaped like (nr, nz), call like `psi = flux_solver(rhs)`
        to get `psi` in [Wb] or [V-s], where `rhs = -2.0 * np.pi * mu_0 * rmesh * jtor` with the boundary
        values set to the circular-filament solved flux.
        """
        return flux_solver(self.grids)

    def get_coil_names(self) -> list[str]:
        """Get coil names in the same order as their indices"""
        # Leaving this function in for backwards compatibility
        return self.coil_names

    @cached_property
    def coil_names(self) -> list[str]:
        """Coil names in the same order as their indices"""
        return [c.name for c in self.coils]

    def get_coil_index_dict(self) -> dict[str, int]:
        """Get a map from coil names to indices in tables and self.coils"""
        # Leaving this function in for backwards compatibility
        return self.coil_index_dict

    @cached_property
    def coil_index_dict(self) -> dict[str, int]:
        """A map from coil names to indices in tables and self.coils"""
        return {self.coils[i].name: i for i in range(len(self.coils))}

    def calc_plasma_flux(
        self,
        current_density: NDArray[F64],
        calc_method: Literal["table", "solve"] = "table",
    ) -> NDArray:
        """
        Calculate the flux field associated with a given plasma current density distribution,
        either by summing over baked flux tables or by solving the Grad-Shafranov PDE.

        This calculation is most commonly used for the plasma, but is in fact more general,
        and applies to anything with an equivalent toroidal current density and axisymmetry.

        Args:
            current_density: [A/m^2], shape (nr, nz), toroidal current density on finite-difference mesh
            calc_method: Whether to sum over baked tables (memory-intensive, but reliable)
                         or perform Grad-Shafranov linear solve (lean, but fragile). Defaults to "table".

        Returns:
            poloidal flux field, [Wb] with shape (nr, nz)
        """
        assert (
            current_density.shape == self.meshes[0].shape
        ), "Supplied current density shape does not match tables"
        if calc_method == "table":
            dr, dz = self.dxgrid  # [m] grid discretization
            area = dr * dz  # [m^2] cross-sectional area of grid cell
            psi = np.zeros_like(current_density)  # [Wb]
            for i, jtor_cell in enumerate(current_density.flatten()):  # [A/m^2]
                psi_table_part = self.plasma_flux_tables[i, :, :]  # [Wb/A]
                current = jtor_cell * area  # [A]
                psi += current * psi_table_part  # [Wb]
        elif calc_method == "solve":
            psi = solve_flux_axisymmetric(
                self.grids, self.meshes, current_density, self.flux_solver
            )

        return psi  # [Wb]

    def calc_plasma_flux_density(
        self,
        plasma_flux: NDArray[F64],
    ) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        Calculate a 4th-order estimate of the plasma's B-field via finite
        difference on the flux field.

        Args:
            plasma_flux: [Wb] with shape (nr, nz), Solved (or summed) plasma poloidal flux

        Returns:
            (br, bz) [T/A] with shape (nr, nz), plasma flux density (B-field)
        """
        return calc_flux_density_from_flux(plasma_flux, *self.meshes)

    def calc_plasma_self_inductance(
        self,
        plasma_current: float,
        plasma_poloidal_flux: NDArray[F64],
        br_plasma: NDArray[F64],
        bz_plasma: NDArray[F64],
        plasma_surface: tuple[NDArray[F64], NDArray[F64]],
        plasma_mask: NDArray[F64],
    ) -> tuple[float, float, float]:
        """
        Calculate the plasma's instantaneous poloidal self-inductance from two components
        addressing the energy stored inside and outside the plasma volume, returning both
        components and the total because the components are sometimes used for other calcs.

        See `self.calc_plasma_flux()`, `self.calc_plasma_flux_density()`, and
        `device_inductance.contour.trace_contour()` for methods to build the inputs
        to this function.

        This method is of marginal use on its own; because any change in current in
        the inductive system will result in motion of the plasma current distribution,
        which changes the plasma self-inductance, the instantaneous self-inductance only
        provides one of two terms in determining the loop voltage response of the plasma.

        With a system of just the plasma and examining only inductive voltage for clarity:

        ```
        V_loop = I*(dL/dt) + L*(dI/dt) = I*(dL/dI)*(dI/dt) + L*(dI/dt)
                             ^                ^
                             |                |
           This formula's L  -                - This quantity is also needed
                                                for a linear treatment
        ```

        This extends to coupled systems with multiple inductors in a similar way.

        Details of the method used can be found in the
        [docs for cfsem.](https://cfsem-py.readthedocs.io/en/latest/python/inductance/#cfsem.self_inductance_distributed_axisymmetric_conductor)

        Args:
            plasma_current: [A] total toroidal current in the plasma
            plasma_poloidal_flux: [Wb] solved plasma flux field
            br_plasma: [T] solved plasma magnetic flux density, R-component
            bz_plasma: [T] solved plasma magnetic flux density, Z-component
            plasma_surface: [m] r,z coordinates of plasma last closed flux surface (aka LCFS, aka bounding contour)
            plasma_mask: [dimensionless] binary mask of plasma interior points. 1 inside, 0 outside.

        Returns:
            (Lt, Li, Le) [H] total, internal, and external instantaneous self-inductances
        """
        total_inductance, internal_inductance, external_inductance = (
            self_inductance_distributed_axisymmetric_conductor(
                current=plasma_current,
                grid=self.grids,
                mesh=self.meshes,
                b_part=(br_plasma, bz_plasma),
                psi_part=plasma_poloidal_flux,
                mask=plasma_mask,
                edge_path=plasma_surface,
            )
        )

        return total_inductance, internal_inductance, external_inductance  # [H]

    @cached_property
    def _calc_model_reduction(self) -> tuple[NDArray[F64], NDArray[F64], int]:
        """Internal function for generating model-reduction outputs"""
        if self.model_reduction_method == "eigenmode":
            d, tuv, neig = model_reduction.eigenmode_reduction(
                self.structure_mutual_inductances,
                self.structure_resistances,
                self.max_nmodes,
            )
        elif self.model_reduction_method == "stabilized eigenmode":
            d, tuv, neig = model_reduction.stabilized_eigenmode_reduction(
                self.structure_mutual_inductances,
                self.structure_resistances,
                self.max_nmodes,
            )
        else:
            raise ValueError(
                f"Unrecognized model reduction method `{self.model_reduction_method}`"
            )

        return d, tuv, neig


@dataclass(frozen=True)
class TypicalOutputs:
    """
    A fully-actualized (precomputed) set of matrices and tables
    covering typical usage patterns.
    """

    device: DeviceInductance
    """
    Thin wrapper to provide methods and properties
    on a device description ODS.
    """

    extent: tuple[float, float, float, float]
    """
    [m] Extent of the computational domain's cell centers.
    This will be updated during initialization, during which it
    may be adjusted to satisfy the required spatial resolution.
    """
    extent_for_plotting: tuple[float, float, float, float]
    """
    [m] The full extent of the computational domain implied by
    the cell centers, expanded a half-cell to the boundary cells' edges
    like [extent[0] - dr/2, extent[1] + dr/2, ...]
    """

    dxgrid: tuple[float, float]
    """[m] spatial resolution of computational grid"""

    meshes: tuple[NDArray[F64], NDArray[F64]]
    """[m] with shape ((nr, nz), (nr, nz)), 2D meshgrids over domain"""
    grids: tuple[NDArray[F64], NDArray[F64]]
    """[m] with shape ((nr), (nz)), 1D grids"""

    nmodes: int
    """Number of structure modes retained"""

    mcc: NDArray[F64]
    """[H] with shape (ncoil X ncoil), Coil-coil mutual inductance"""
    mss: NDArray[F64]
    """[H] with shape (nstruct X nstruct), Structure-structure mutual inductance"""
    mcs: NDArray[F64]
    """[H] with shape (ncoil X nstruct), Coil-structure mutual inductance"""

    r_c: NDArray[F64]
    """[ohm] with shape (ncoil X ncoil), Resistance of coils as diagonal matrix"""
    r_s: NDArray[F64]
    """[ohm] with shape (nstruct X nstruct), Resistance of structure fils, diagonal"""
    r_modes: NDArray[F64]
    """[ohm] with shape (neig X neig), Resistance of structure modes"""

    tuv: NDArray[F64]
    """
    [dimensionless] with shape (nstruct X neig), structure model-reduction matrix.\n
    Each column is a normalized vector of unit length.

    Apply this transform like `a_transformed = a @ tuv` for matrices
    with `a.shape == (..., nstruct)`.
    In the case of diagonalized structure filament resistances,
    the transform is applied like `r_transformed = tuv.T @ (r @ tuv)`.
    """

    psi_c: NDArray[F64]
    """[Wb/A] with shape (ncoils X nr X nz), Coil flux tables"""
    psi_s: NDArray[F64]
    """[Wb/A] with shape (nstruct X nr X nz), Structure flux tables"""
    psi_modes: NDArray[F64]
    """[Wb/A] with shape (nmodes X nr X nz), Structure mode flux tables"""
