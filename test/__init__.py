import pytest

import device_inductance


@pytest.fixture(scope="session")
def typical_outputs() -> device_inductance.TypicalOutputs:
    # Set up a regular computational grid
    # It would be nice to use a coarser grid,
    # but in order for the tables to be testable,
    # we need a fairly fine one
    dxgrid = (0.05, 0.04)  # Different resolution to make sure they are never swapped
    extent = (2.0 * dxgrid[0], 4.5, -3.0, 3.0)

    # Load the default device
    ods = device_inductance.load_default_ods()

    # Pre-compute the usual set of matrices and tables
    typical_outputs = device_inductance.typical(
        ods, extent, dxgrid, max_nmodes=40, show_prog=False
    )

    return typical_outputs


@pytest.fixture(scope="session")
def typical_outputs_stabilized_eigenmode() -> device_inductance.TypicalOutputs:
    # Because the device is immutable after init, we have to make a whole new one
    # to get the other model reduction method

    # Set up a regular computational grid
    # It would be nice to use a coarser grid,
    # but in order for the tables to be testable,
    # we need a fairly fine one
    dxgrid = (0.05, 0.04)  # Different resolution to make sure they are never swapped
    extent = (2.0 * dxgrid[0], 4.5, -3.0, 3.0)

    # Load the default device
    ods = device_inductance.load_default_ods()

    # Pre-compute the usual set of matrices and tables
    typical_outputs = device_inductance.typical(
        ods,
        extent,
        dxgrid,
        max_nmodes=40,
        show_prog=False,
        model_reduction_method="stabilized eigenmode",
    )

    return typical_outputs
