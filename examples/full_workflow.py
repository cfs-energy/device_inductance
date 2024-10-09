"""End-to-end workflow generating all outputs and some exploratory plots"""

import numpy as np
import matplotlib.pyplot as plt

import device_inductance

# Don't spam the terminal if this is running on a build server
show_prog = plt.get_backend().lower() != "agg"

# Set up a regular computational grid
# The extent defined here will be adjusted during init to respect the required resolution
dxgrid = (0.05, 0.05)  # [m] grid resolution
dr, dz = dxgrid
extent = (dr * 2.0, 4.5, -3.0, 3.0)

# Load the default device
ods = device_inductance.load_default_ods()

# Pre-compute the usual set of matrices and tables
print("Generating typical outputs\n")
typical_outputs = device_inductance.typical(ods, extent, dxgrid, show_prog=show_prog)

# Pull out the full-featured device object
# which can be used to generate anything not in the typical workflow outputs
device = typical_outputs.device

# Populate some tables which are not part of the typical outputs
# and will be calculated on first access
print("\nGenerating optional outputs\n")
psi_mesh_plasma = device.plasma_flux_tables
br_coils, bz_coils = device.coil_flux_density_tables
br_structures, bz_structures = device.structure_flux_density_tables
br_modes, bz_modes = device.structure_mode_flux_density_tables
m_circuit_circuit = device.circuit_mutual_inductances
m_circuit_structure = device.circuit_structure_mutual_inductances
m_circuit_mode = device.circuit_structure_mode_mutual_inductances
psi_circuits = device.circuit_flux_tables
br_circuits, bz_circuits = device.circuit_flux_density_tables

# Get the extent and mesh to use for plots
extent = typical_outputs.extent_for_plotting
ar = (extent[3] - extent[2]) / (extent[1] - extent[0])  # aspect ratio of mesh

# Make some validation plots
ncoil = device.n_coils
ncirc = device.n_circuits
nmodes = typical_outputs.nmodes


def table_imshow(arr, contours=True):
    """Helper function of plotting flux and B-field tables"""
    cmap = "seismic"
    vmax = np.max(np.abs(arr))
    vmin = -vmax
    plt.imshow(
        arr,
        extent=extent,
        origin="lower",
        interpolation="bicubic",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if contours:
        _cs = plt.contour(
            arr,
            extent=extent,
            levels=4,
            colors="k",
            linewidths=1,
        )


plt.figure(figsize=(5, 6.5))
structure_rs = [x.r for x in device.structures]
structure_zs = [x.z for x in device.structures]
areas = np.array([x.area for x in device.structures])
mask_for_plot = 1.0 - device.limiter_mask.copy()
plt.imshow(
    device.limiter_mask.T,
    origin="lower",
    extent=extent,
    interpolation="nearest",
    cmap="BuGn",
)
plt.scatter(
    structure_rs,
    structure_zs,
    s=5 * areas / np.max(areas),
    marker=".",
    color="k",
    alpha=1,
)
for s in device.structures:
    plt.plot(*s.polygon.boundary.xy, color="k")
for c in device.coils:
    coil_rs = [f.r for f in c.filaments]
    coil_zs = [f.z for f in c.filaments]
    plt.scatter(coil_rs, coil_zs, color="k")
plt.plot(*device.limiter.boundary.xy, color="k")
plt.gca().axis("equal")
plt.title("Filaments and Structure Mesh")
plt.xlabel("R [m]")
plt.ylabel("Z [m]")
plt.tight_layout()

# plt.figure()  # matshow makes its own figure
plt.matshow(typical_outputs.mcc)
plt.colorbar()
plt.title("Coil-Coil Mutual Inductances [H]")

plt.matshow(m_circuit_circuit)
plt.colorbar()
plt.title("Circuit-Circuit Mutual Inductances [H]")

plt.figure()
plt.imshow(
    m_circuit_structure,
    extent=[0, 1, 0, 1],
    origin="lower",
    interpolation="nearest",
)
plt.colorbar()
plt.title("Circuit-Structure Mutual Inductances [H]")

plt.matshow(m_circuit_mode)
plt.colorbar()
plt.title("Circuit-Mode Mutual Inductances [H]")

plt.matshow(typical_outputs.mss)
plt.colorbar()
plt.title("Passive Structure Mutual Inductances [H]")

plt.figure()
plt.imshow(
    typical_outputs.mcs,
    extent=[0, 1, 0, 1],
    origin="lower",
    interpolation="nearest",
)
plt.colorbar()
plt.title("Coil-Structure Mutual Inductances [H]")

plt.figure()
plt.imshow(
    typical_outputs.tuv, extent=[0, 1, 0, 1], origin="lower", interpolation="nearest"
)
d = device.structure_mode_eigenvalues
plt.plot(
    np.linspace(0.0, 1.0, nmodes, endpoint=True),
    np.abs(d / d[0]),
    color="k",
    label="Normalized Eigenvalues",
)  # eigenvalues
plt.legend()
plt.colorbar()
plt.title(f"Tuv, Structure Model Reduction Transform\n(neig = {nmodes})")

n = int(nmodes**0.5)
m = nmodes // n
n = n if n * m >= nmodes else n + 1
fig, axes = plt.subplots(n, m, figsize=(min(n, 8), min(m, 8)))
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(nmodes):
    plt.sca(axes[i])
    table_imshow(
        typical_outputs.psi_modes[i].T,
    )
    plt.scatter(structure_rs, structure_zs, s=5, marker=".", color="k", alpha=1)
    plt.title(f"Mode {i}")
for ax in axes:
    plt.sca(ax)
    plt.axis("off")
plt.suptitle("Eigenmode Flux Maps")

n = int(nmodes**0.5)
m = nmodes // n
n = n if n * m >= nmodes else n + 1
fig, axes = plt.subplots(n, m, figsize=(min(n, 8), min(m, 8)))
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(nmodes):
    plt.sca(axes[i])
    table_imshow(br_modes[i].T)
    plt.scatter(structure_rs, structure_zs, s=5, marker=".", color="k", alpha=1)
    plt.title(f"Mode {i}")
for ax in axes:
    plt.sca(ax)
    plt.axis("off")
plt.suptitle("Eigenmode Br Maps")

n = int(nmodes**0.5)
m = nmodes // n
n = n if n * m >= nmodes else n + 1
fig, axes = plt.subplots(n, m, figsize=(min(n, 8), min(m, 8)))
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(nmodes):
    plt.sca(axes[i])
    table_imshow(bz_modes[i].T)
    plt.scatter(structure_rs, structure_zs, s=5, marker=".", color="k", alpha=1)
    plt.title(f"Mode {i}")
for ax in axes:
    plt.sca(ax)
    plt.axis("off")
plt.suptitle("Eigenmode Bz Maps")

plt.figure()
plt.title("Sum of Passive Filament Flux Maps")
table_imshow(np.sum(typical_outputs.psi_s, axis=0).T)
plt.scatter(
    structure_rs,
    structure_zs,
    s=5 * areas / np.max(areas),
    marker=".",
    color="k",
    alpha=1,
)


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
plt.suptitle("Sum of Passive Structure Filament B-field Maps")
plt.sca(axes[0])
table_imshow(np.sum(br_structures, axis=0).T)
plt.scatter(
    structure_rs,
    structure_zs,
    s=5 * areas / np.max(areas),
    marker=".",
    color="k",
    alpha=1,
)
plt.title("Br")
plt.sca(axes[1])
table_imshow(np.sum(bz_structures, axis=0).T)
plt.scatter(
    structure_rs,
    structure_zs,
    s=5 * areas / np.max(areas),
    marker=".",
    color="k",
    alpha=1,
)
plt.title("Bz")


n = int(ncoil**0.5)
m = ncoil // n
n = n if n * m >= ncoil else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Coil Flux Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(ncoil):
    coil = device.coils[i]
    plt.sca(axes[i])
    table_imshow(typical_outputs.psi_c[i].T)
    plt.scatter(
        [x.r for x in coil.filaments],
        [x.z for x in coil.filaments],
        marker=".",
        color="k",
        s=5,
    )
    plt.title(coil.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")

n = int(ncoil**0.5)
m = ncoil // n
n = n if n * m >= ncoil else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Coil Br Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(ncoil):
    coil = device.coils[i]
    plt.sca(axes[i])
    table_imshow(br_coils[i].T)
    plt.scatter(
        [x.r for x in coil.filaments],
        [x.z for x in coil.filaments],
        marker=".",
        color="k",
        s=5,
    )
    plt.title(coil.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")

n = int(ncoil**0.5)
m = ncoil // n
n = n if n * m >= ncoil else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Coil Bz Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for i in range(ncoil):
    coil = device.coils[i]
    plt.sca(axes[i])
    table_imshow(bz_coils[i].T)
    plt.scatter(
        [x.r for x in coil.filaments],
        [x.z for x in coil.filaments],
        marker=".",
        color="k",
        s=5,
    )
    plt.title(coil.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")


n = int(ncirc**0.5)
m = ncirc // n
n = n if n * m >= ncirc else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Circuit Flux Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for j in range(ncirc):
    plt.sca(axes[j])
    circuit = device.circuits[j]
    table_imshow(psi_circuits[j].T)
    for i in [c[0] for c in circuit.coils]:
        coil = device.coils[i]
        plt.scatter(
            [x.r for x in coil.filaments],
            [x.z for x in coil.filaments],
            marker=".",
            color="k",
            s=5,
        )
        plt.title(circuit.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")


n = int(ncirc**0.5)
m = ncirc // n
n = n if n * m >= ncirc else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Circuit Br Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for j in range(ncirc):
    plt.sca(axes[j])
    circuit = device.circuits[j]
    table_imshow(br_circuits[j].T)
    for i in [c[0] for c in circuit.coils]:
        coil = device.coils[i]
        plt.scatter(
            [x.r for x in coil.filaments],
            [x.z for x in coil.filaments],
            marker=".",
            color="k",
            s=5,
        )
        plt.title(circuit.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")

n = int(ncirc**0.5)
m = ncirc // n
n = n if n * m >= ncirc else n + 1
fig, axes = plt.subplots(n, m, figsize=(6 * n / m / ar, ar * 8 * m / n))
plt.suptitle("Circuit Bz Maps")
axes = sum([[y for y in x] for x in axes], start=[])
for j in range(ncirc):
    plt.sca(axes[j])
    circuit = device.circuits[j]
    table_imshow(bz_circuits[j].T)
    for i in [c[0] for c in circuit.coils]:
        coil = device.coils[i]
        plt.scatter(
            [x.r for x in coil.filaments],
            [x.z for x in coil.filaments],
            marker=".",
            color="k",
            s=5,
        )
        plt.title(circuit.name)
for ax in axes:
    plt.sca(ax)
    plt.axis("off")

plt.show()
