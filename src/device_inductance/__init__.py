from importlib.metadata import metadata
from pathlib import Path
from typing import Literal

__version__ = metadata(str(__package__))["Version"]

from omas import ODS, load_omas_json

from device_inductance import (
    contour,
    logging,
    mesh,
    model_reduction,
    sensors,
    structures,
)
from device_inductance.coils import Coil, CoilFilament
from device_inductance.device import DeviceInductance, TypicalOutputs
from device_inductance.grid import Extent, GridSpec, Resolution
from device_inductance.logging import log, logger_is_set_up, logger_setup_default
from device_inductance.structures import PassiveStructureLoop
from device_inductance.utils import (
    calc_flux_density_from_flux,
    flux_solver,
    gradient_order4,
    solve_flux_axisymmetric,
)


def load_default_ods() -> ODS:
    """
    Load an example ODS file in the format required by device_inductance.

    The example differs from real SPARC configurations in at least the following ways:
      * Coil number of turns and resistances are obfuscated
      * Actual magnetics sensors are replaced with mockup examples
      * The example description may be arbitrarily out-of-date, as it is not updated regularly
    """

    if not logger_is_set_up():
        logger_setup_default()

    log().warning("""
Loading example device description.
The example differs from real SPARC configurations in at least the following ways:
    * Coil number of turns and resistances are obfuscated
    * Actual magnetics sensors are replaced with mockup examples
    * The example description may be arbitrarily out-of-date, as it is not updated regularly
                  """)

    # NOTE: This should be rewritten to use importlib once omas supports loading raw text
    ods_filename = Path(__file__).parent / "../examples/OS_SPARC_Device_Description.json"
    with open(ods_filename) as f:
        ods = load_omas_json(f)

    # Add some dummy sensors to exercise the sensor functions
    #   Full flux loop
    ffloop = ods["magnetics.flux_loop.0"]
    ffloop["type.index"] = 1
    ffloop["name"] = "dummy_full_flux_loop"
    ffloop["position.0.r"] = 3.0
    ffloop["position.0.z"] = 0.0
    #   Partial flux loop
    pfloop = ods["magnetics.flux_loop.1"]
    pfloop["type.index"] = 2
    pfloop["name"] = "dummy_partial_flux_loop"
    pfloop["position.0.r"] = 3.0
    pfloop["position.0.phi"] = 0.0
    pfloop["position.0.z"] = 0.0
    pfloop["position.1.r"] = 3.3
    pfloop["position.1.phi"] = 0.25
    pfloop["position.1.z"] = 1.0
    #   Bpol probe
    bpcoil = ods["magnetics.b_field_pol_probe.0"]
    bpcoil["type.index"] = 2
    bpcoil["name"] = "dummy_poloidal_b_field_probe"
    bpcoil["position.r"] = 3.1
    bpcoil["position.phi"] = 0.0
    bpcoil["position.z"] = 0.5
    bpcoil["poloidal_angle"] = 0.7

    return ods


def typical(
    ods: ODS,
    min_extent: Extent | None = None,
    dxgrid: Resolution = (0.0, 0.0),
    max_nmodes: int = 40,
    model_reduction_method: Literal["eigenmode", "stabilized eigenmode"] = "eigenmode",
    show_prog: bool = True,
    plasma_coil_force_method: Literal["tables", "mask"] = "mask",
    n_radial_slices: int = 30,
    gridspec: GridSpec | None = None,  # Appended to avoid breaking change
    **kwargs,  # For backwards compatibility with `extent` kwarg only
) -> TypicalOutputs:
    """
    Generate a typical set of outputs,
    notably excluding the plasma flux tables which usually require much more
    run time than the rest of the outputs combined.

    Note: during initialization, the extent of the computational grid may be
    adjusted to achieve the target spatial resolution. The adjusted extent
    will always bound the requested extent.

    Args:
        ods: An OMAS object in the format produced by device_description
        min_extent: [m] rmin, rmax, zmin, zmax extent of computational domain.
                    This will be updated during mesh initialization, during which it
                    may be adjusted to satisfy the required spatial resolution.
        dxgrid: [m] Spatial resolution of computational grid
        max_nmodes: Maximum number of structure modes to keep. Defaults to 40.
        show_prog: Whether to show terminal progress bars. Defaults to True.
        plasma_coil_force_method: Whether to interpolate B-field on the fully-realized mesh tables,
                                  or do direct filament calculations from points inside the limiter mask.
                                  Defaults to "mask", which is faster and uses less memory, but only includes
                                  nonzero entries inside the limiter, which requires a valid limiter geometry.
        n_radial_slices: Number of radial slices to use for chunking large structures. Each slice is centered
                            at the limiter centroid.
        gridspec: Exact alternative to min_extent. Only one of min_extent or gridspec should be provided.

    Returns:
        A fully-computed set of matrices and tables covering the needs of a typical workflow
    """
    if not logger_is_set_up():
        logger_setup_default()

    if "extent" in kwargs:
        # Backwards compatibility with `extent` kwarg name only
        log().warning("`extent` input is deprecated; use `min_extent` or `gridspec` instead")
        min_extent = kwargs.pop("extent")

    device = DeviceInductance(
        ods=ods,
        max_nmodes=max_nmodes,
        min_extent=min_extent,
        gridspec=gridspec,
        dxgrid=dxgrid,
        model_reduction_method=model_reduction_method,
        show_prog=show_prog,
        plasma_coil_force_method=plasma_coil_force_method,
        n_radial_slices=n_radial_slices,
    )

    out = TypicalOutputs(
        device=device,
        extent=device.extent,
        dxgrid=device.dxgrid,
        meshes=device.meshes,
        grids=device.grids,
        extent_for_plotting=device.extent_for_plotting,
        mcc=device.coil_mutual_inductances,
        mss=device.structure_mutual_inductances,
        mcs=device.coil_structure_mutual_inductances,
        r_s=device.structure_resistances,
        r_c=device.coil_resistances,
        r_modes=device.structure_mode_resistances,
        tuv=device.structure_model_reduction,
        nmodes=device.n_structure_modes,
        psi_c=device.coil_flux_tables,
        psi_s=device.structure_flux_tables,
        psi_modes=device.structure_mode_flux_tables,
    )
    return out


__all__ = [
    "DeviceInductance",
    "TypicalOutputs",
    "typical",
    "Coil",
    "CoilFilament",
    "PassiveStructureLoop",
    "load_default_ods",
    "calc_flux_density_from_flux",
    "gradient_order4",
    "flux_solver",
    "solve_flux_axisymmetric",
    "logging",
    "model_reduction",
    "contour",
    "sensors",
    "mesh",
    "structures",
]
