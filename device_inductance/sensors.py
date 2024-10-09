from dataclasses import dataclass
from warnings import warn
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from interpn import MulticubicRectilinear

from omas import ODS


@dataclass(frozen=True)
class PoloidalFieldProbe:
    """
    A poloidal B-field sensor at a point in space; aka Mirnov probe or pickup coil.
    Idealized here as attached to an ideal integrator s.t. it senses B integrated
    in time instead of rate of change.

    In reality, the hardware senses dB/dt on some vector in the poloidal plane,
    but this is not so useful in a digital context without high-bandwidth integration.
    """

    name: str
    """Sensor name"""

    r: float
    """[m] r-location"""
    phi: float
    """[rad] angular location (about z)"""
    z: float
    """[m] z-location"""

    poloidal_angle: float
    """[rad] poloidal orientation (about phi), clockwise from +R-axis"""

    @property
    def unit_vector(self) -> NDArray:
        """
        Get the unit vector in the rz plane that the measurement is projected on.
        Uses cocos-11 coordinate system, with poloidal angle clockwise about into-the-board
        toroidal direction, starting from the outboard (r=1, z=0) direction.
        """
        z = -np.sin(self.poloidal_angle)
        r = np.cos(self.poloidal_angle)

        return np.atleast_1d(np.array((r, z)))

    def response(
        self, grids: tuple[NDArray, NDArray], br: NDArray, bz: NDArray
    ) -> float:
        """Ideal integrated response to a given local B-field.  This is the sum of local B-field
        components, not the normed field!

        Args:
            grids: [m] 1D arrays describing rectilinear mesh
            br: [T]
            bz: [T]

        Raises:
            ValueError: If the sensor is not inside the mesh

        Returns:
            [T] ideal integrated B-field, sum of components projected on the axis of the probe
        """

        # Build interpolators - this is fast but not totally free
        br_interpolator = MulticubicRectilinear.new(
            [x for x in grids], br.flatten(), linearize_extrapolation=True
        )
        bz_interpolator = MulticubicRectilinear.new(
            [x for x in grids], bz.flatten(), linearize_extrapolation=True
        )

        # Unpack/repack geometry
        unit_vector = self.unit_vector
        point_to_interp = [np.atleast_1d(self.r), np.atleast_1d(self.z)]

        # Check if we are extrapolating the fields
        out_of_bounds_flags = br_interpolator.check_bounds(point_to_interp, atol=1e-3)
        if np.any(out_of_bounds_flags):
            raise ValueError(
                f"Poloidal field probe {self.name} outside of mesh bounds."
                + f" Mesh bounds flags: {out_of_bounds_flags}"
            )

        # Interpolate fields
        br_interped = br_interpolator.eval(point_to_interp)[0]  # [T]
        bz_interped = bz_interpolator.eval(point_to_interp)[0]

        # Aggregate response
        # This is the sum of B components aligned with direction of probe;
        # this is _not_ the normed magnitude, it's a sum, because the real probe integrates
        # flux components not normed field
        br_projected = br_interped * unit_vector[0]  # [T]
        bz_projected = bz_interped * unit_vector[1]
        b_projected = br_projected + bz_projected  # [T]

        # The response is based on an ideal integrator, so we get the
        # exact B-field instead of a voltage
        response = b_projected  # [T]

        return response  # [T]


@dataclass(frozen=True)
class FullFluxLoop:
    """
    Flux sensor wrapped around the entire toroid, encompassing all the
    poloidal projected area from this point inward.
    Idealized here as attached to an ideal integrator s.t. it senses
    flux integrated in time instead of rate of change.
    ::

                , - ~ ~ ~ - ,
            , '               ' ,
          ,                       ,         phi
         ,         A_pol           ,         ^
        ,                           ,        |
        ,              -----------> * p1     x
        ,                    R      ,        Z
         ,                         ,
          ,                       ,
            ,                  , '
              ' - , _ _ _ ,  '
                     ||
                    _||_
                    -  +
    """

    name: str
    """Sensor name"""

    r: float
    """[m] r-location"""
    z: float
    """[m] z-location"""

    def response(self, grids: tuple[NDArray, NDArray], psi: NDArray) -> float:
        """Ideal integrated response to a given flux field

        Args:
            grids: [m] 1D arrays describing rectilinear mesh
            psi: [Wb], 2D array of rate of change of poloidal flux

        Raises:
            ValueError: If the sensor is not inside the mesh

        Returns:
            [Wb] ideal integrated response
        """
        # Build interpolators - this is fast but not totally free
        psi_interpolator = MulticubicRectilinear.new(
            [x for x in grids], psi.flatten(), linearize_extrapolation=True
        )  # [V], from [Wb/s] or [V-s/s]

        # Unpack/repack geometry
        point_to_interp = [np.atleast_1d(self.r), np.atleast_1d(self.z)]

        # Check if we are extrapolating the fields
        out_of_bounds_flags = psi_interpolator.check_bounds(point_to_interp, atol=1e-3)
        if np.any(out_of_bounds_flags):
            raise ValueError(
                f"Full flux loop {self.name} outside of mesh bounds."
                + f" Mesh bounds flags: {out_of_bounds_flags}"
            )

        # Interpolate field
        psi_interped = psi_interpolator.eval(point_to_interp)[0]  # [Wb]

        # The response assumes an ideal integrator, so we get a flux value instead of a voltage
        response = psi_interped  # [Wb]

        return response  # [Wb]


@dataclass(frozen=True)
class PartialFluxLoop:
    """
    Flux sensor defined in terms of the opposite corners of a rectangle
    wrapped part-way around the toroid, enclosing some poloidal projected area
    but no toroidal projected area.
    Idealized here as attached to an ideal integrator s.t. it senses
    flux integrated in time instead of rate of change.

    ::

                                   Z
           * --------- * p2        ^
           |           |           |
           |           |           x ---> phi
        p1 * --------- *           R

    The normal vector is taken as clockwise from the vector formed by the first and second points.

        o (r1, z1)
        |
        |------> normal
        |
        |
        o (r0, z0)
    """

    name: str
    """Sensor name"""

    r0: float
    """[m] r-location of first corner"""
    phi0: float
    """[rad] angular location (about z) of first corner"""
    z0: float
    """[m] z-location of first corner"""

    r1: float
    """[m] r-location of second corner"""
    phi1: float
    """[rad] angular location (about z) of first corner"""
    z1: float
    """[m] z-location of second corner"""

    n_discretization: int = 1000
    """
    Number of points to use for integrating the flux field.
    Meant to be a constant that is tuned manually to target an acceptable error level.
    """

    @property
    def normal_vector(self) -> NDArray:
        """
        [dimensionless] (r, z) components of the normal vector
        of the surface enclosed by the loop
        """
        dr = self.r1 - self.r0  # [m]
        dz = self.z1 - self.z0  # [m]
        norm = (dr**2 + dz**2) ** 0.5  # [m]

        normal = np.array((dz, -dr)) / norm  # [dimensionless]
        return normal  # [dimensionless] clockwise normal vector

    @property
    def loop_frac(self) -> float:
        """Portion of the full toroid that is swept by this loop"""
        return abs(self.phi1 - self.phi0) / (2.0 * np.pi)  # [dimensionless]

    @cached_property
    def discretized_rz_section(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Generate evenly-sampled points along the r-z section of the loop and
        the projected area associated with each point.

        Because the projected area at each point is different, 1D trapezoid
        integration is inconvenient; instead, points are sampled entirely
        inside the span of the section, not including the endpoints, in order
        to support a Riemann sum over dot(B, dA) using a finer resolution.
        """
        dr = self.r1 - self.r0  # [m]
        dz = self.z1 - self.z0  # [m]
        norm = (dr**2 + dz**2) ** 0.5  # [m] length of cross-section
        length_per_segment = norm / float(self.n_discretization)  # [m]
        weights = np.linspace(0.0, 1.0, self.n_discretization + 2, endpoint=True)[
            1:-1
        ]  # [dimensionless]
        rs = self.r0 + dr * weights  # [m]
        zs = self.z0 + dz * weights  # [m]
        das = 2.0 * np.pi * rs * length_per_segment * self.loop_frac  # [m^2]

        return rs, zs, das  # [m],[m] sampled points, [m^2] projected area at each point

    def response(
        self, grids: tuple[NDArray, NDArray], br: NDArray, bz: NDArray
    ) -> float:
        """Ideal integrated response to a given local B-field

        Args:
            grids: [m] 1D arrays describing rectilinear mesh
            br: [T]
            bz: [T]

        Raises:
            ValueError: If the sensor is not inside the mesh

        Returns:
            [Wb] ideal integrated flux through the loop
        """

        # Build interpolators - this is fast but not totally free
        br_interpolator = MulticubicRectilinear.new(
            [x for x in grids], br.flatten(), linearize_extrapolation=True
        )
        bz_interpolator = MulticubicRectilinear.new(
            [x for x in grids], bz.flatten(), linearize_extrapolation=True
        )

        # Unpack/repack geometry
        rs, zs, das = self.discretized_rz_section  # [m], [m], [m^2]
        points_to_interp = [rs, zs]  # [m]
        normal = self.normal_vector  # [dimensionless]

        # Check if we are extrapolating the fields
        out_of_bounds_flags = br_interpolator.check_bounds(points_to_interp, atol=1e-3)
        if np.any(out_of_bounds_flags):
            raise ValueError(
                f"Partial flux loop {self.name} outside of mesh bounds."
                + f" Mesh bounds flags: {out_of_bounds_flags}"
            )

        # Interpolate fields
        br_interped = br_interpolator.eval(points_to_interp)  # [T]
        bz_interped = bz_interpolator.eval(points_to_interp)

        # Aggregate response
        # This is the sum of B components aligned with direction of probe;
        # this is _not_ the normed magnitude, it's a sum, because the real probe integrates
        # flux components not normed field
        br_projected = br_interped * normal[0]  # [T]
        bz_projected = bz_interped * normal[1]
        b_projected = br_projected + bz_projected  # [T]

        # The response assumes an ideal integrator, so we get the integrated flux instead of a voltage
        response = float(np.sum(b_projected * das))  # [Wb]

        return response  # [Wb]


def _extract_full_flux_loops(description: ODS) -> list[FullFluxLoop]:
    """
    Extract information about full flux loops.

    Args:
        description: Device geometric info in the format produced by device_description

    Returns:
        A list of full flux loop objects
    """
    loops = []

    ods_flux_loops = description["magnetics.flux_loop"]
    for ods_loop in ods_flux_loops.values():
        if ods_loop["type.index"] != 1:  # Type 1 is full loop
            continue
        loop = FullFluxLoop(
            name=ods_loop["name"],
            r=ods_loop["position.0.r"],
            z=ods_loop["position.0.z"],
        )
        loops.append(loop)

    return loops


def _extract_partial_flux_loops(description: ODS) -> list[PartialFluxLoop]:
    """
    Extract information about partial ("saddle") flux loops.

    Args:
        description: Device geometric info in the format produced by device_description

    Returns:
        A list of partial flux loop objects
    """
    loops = []

    ods_flux_loops = description["magnetics.flux_loop"]
    for ods_loop in ods_flux_loops.values():
        if ods_loop["type.index"] != 2:  # Type 2 is partial loop
            continue
        loop = PartialFluxLoop(
            name=ods_loop["name"],
            r0=ods_loop["position.0.r"],
            phi0=ods_loop["position.0.phi"],
            z0=ods_loop["position.0.z"],
            r1=ods_loop["position.1.r"],
            phi1=ods_loop["position.1.phi"],
            z1=ods_loop["position.1.z"],
        )
        loops.append(loop)

    return loops


def _extract_poloidal_field_probes(description: ODS) -> list[PoloidalFieldProbe]:
    """
    Extract information about poloidal field probes.

    Args:
        description: Device geometric info in the format produced by device_description

    Returns:
        A list of poloidal field probe objects
    """
    probes = []

    ods_bps = description["magnetics.b_field_pol_probe"]
    for ods_bp in ods_bps.values():
        bp_type_number = ods_bp["type.index"]
        if bp_type_number != 2:  # Type 2 is Mirnov probe
            warn(
                f"Encountered unimplemented poloidal field probe of type {bp_type_number}; skipping"
            )
            continue
        probe = PoloidalFieldProbe(
            name=ods_bp["name"],
            r=ods_bp["position.r"],
            phi=ods_bp["position.phi"],
            z=ods_bp["position.z"],
            # area_turns=ods_bp["area"],
            poloidal_angle=ods_bp["poloidal_angle"],
        )
        probes.append(probe)

    return probes
