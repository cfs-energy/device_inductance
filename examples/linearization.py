"""
Linearization of the inductive-resistive system including the motion of the plasma in response
to changing coil currents
"""

import matplotlib.pyplot as plt
import numpy as np

import device_inductance
import device_inductance.contour
from device_inductance import DeviceInductance
from device_inductance.linearization import PlasmaLinearization, plasma_response_linearization
from device_inductance.utils import guess_psip

# Don't spam the terminal if this is running on a build server
show_prog = plt.get_backend().lower() != "agg"

# Set up a regular computational grid
dxgrid = (0.05, 0.05)
dr, dz = dxgrid
extent = (dr * 2.0, 4.5, -3.5, 3.5)

# Load the default device
ods = device_inductance.load_default_ods()
device = DeviceInductance(ods=ods, min_extent=extent, dxgrid=dxgrid, show_prog=True)

# Set some arbitrary inputs
ip = 10e6  # [A] 1MA plasma toroidal current just to have a number
rp = 5e-7  # [ohm] plasma loop resistance
actuator_current = 1e3 * np.ones(device.n_circuits)  # [A]
structure_current = 1e3 * np.ones(device.n_structure_modes)  # [A]

# Make an initial guess at the plasma
# which doesn't need to be particularly sane for this example
# but should preferably not interfere with the coil or structure locations
rmesh, zmesh = device.meshes
#    Abuse the psi guess function to make a current density distribution
jshape = guess_psip(rmesh, zmesh, r0=1.8, elongation=1.5) - 0.95
jtor = jshape * np.where(jshape > 0.0, True, False) * device.limiter_mask
jtor *= ip / (np.sum(jtor) * dr * dz)  # [A/m^2] Scale to target plasma current

# Get the plasma flux for this current density
# via tables, because the tables will be used later during the linearization calc,
# although a grad-shafranov solve would be much faster here
psi_p = device.calc_plasma_flux(jtor, calc_method="table")  # [Wb]
br_p, bz_p = device.calc_plasma_flux_density(psi_p)  # [T]

# Trace the last closed flux surface
start = (np.max(rmesh[np.where(jtor)]), 0.0)
plasma_surface = device_inductance.contour.trace_contour(
    device.grids,
    jshape,
    start=start,
    maxis=(2.0, 0.0),
    limiter_circumference=20.0,
    mask_limiter=device.limiter_mask,
)  # [m] path of last closed flux surface

if plasma_surface is None:
    raise ValueError("Contour tracing did not produce a valid plasma surface")

# Calculate plasma self-inductance
lp, _li, _le = device.calc_plasma_self_inductance(
    ip, psi_p, br_p, bz_p, plasma_surface, plasma_mask=np.where(jtor, 1.0, 0.0)
)

# Do the linearization
linearization: PlasmaLinearization = plasma_response_linearization(
    device,
    ip,
    rp,
    lp,
    jtor,
    psi_p,
    actuator_current,
    structure_current,
    actuators="circuits",
    structures="modes",
    gamma_multiplier=1.0,
    include_plasma_in_system=True,
    include_cross_terms=True,
)

print(f"Vertical instability characteristic frequency: {linearization.gamma} [Hz]")
