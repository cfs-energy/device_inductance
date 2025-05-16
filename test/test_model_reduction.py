import numpy as np

import device_inductance

from pytest import approx


from . import typical_outputs, typical_outputs_stabilized_eigenmode, typical_outputs_many_slices  # Required fixture

__all__ = ["typical_outputs", "typical_outputs_stabilized_eigenmode", "typical_outputs_many_slices"]


def test_model_reduction(
    typical_outputs: device_inductance.TypicalOutputs,
    typical_outputs_stabilized_eigenmode: device_inductance.TypicalOutputs,
    typical_outputs_many_slices: device_inductance.TypicalOutputs,
):
    """
    In general,
    dx/dt = Ax + Bu
        y = Cx + Du
    x is current vector I
    u is voltage vector V
    
    For this particular case, with `V - RI = MdI/dt`,
    the state-space reformulation is
    
    dx/dt = dI/dt = -M^-1 @ R @ I + M^-1 @ V
        y =     I = eye @ x + 0 @ u
    
    So
    `A = -M^-1 @ R`
    `B = M^-1`
    `C = eye`
    `D = 0`
    
    We don't need C and D here, because we're not going to use this
    in an actual state space formalization. In fact, the `control`
    library isn't able to simulate even 100us of the full system
    due to the presence of some extremely small timescales.
    
    We also can't easily compare to the `control` library's balred
    (balanced reduction) method here, because the coils have near zero resistance,
    so the system is _almost_ unstable under a constant input, which causes slycot
    (the fortran backend for `control.balred`) balanced reduction to
    sort-of-fail (keeps >200 modes).
    
    Instead, we can check the initial rate of change of coil current under a step in coil voltage
    between the full system and the transformed system."""

    devices = [typical_outputs.device, typical_outputs_stabilized_eigenmode.device, typical_outputs_many_slices.device]
    ndevices = len(devices)

    import matplotlib.pyplot as plt

    _fig, axes = plt.subplots(1, ndevices, sharex=True, sharey=True)

    for iax, device in enumerate(devices):
        # Unpack
        mss = typical_outputs.mss
        mcs = typical_outputs.mcs
        mcc = typical_outputs.mcc

        r_modes = device.structure_mode_resistances
        r_c = typical_outputs.r_c
        r_s = typical_outputs.r_s

        ncoils, nstructs = mcs.shape
        nmodes = device.n_structure_modes

        coils = device.coils

        # Make transformed mutual inductance matrix parts
        mss_transformed = device.structure_mode_mutual_inductances
        mcs_transformed = device.coil_structure_mode_mutual_inductances

        # Make full mutual inductance and resistance matrices
        m = np.vstack((np.hstack((mcc, mcs)), np.hstack((mcs.T, mss))))
        m_transformed = np.vstack(
            (
                np.hstack((mcc, mcs_transformed)),
                np.hstack((mcs_transformed.T, mss_transformed)),
            )
        )

        r = np.diag(np.concatenate((np.diag(r_c), np.diag(r_s)), axis=0))
        r_transformed = np.diag(
            np.concatenate((np.diag(r_c), np.diag(r_modes)), axis=0)
        )

        # Build state-space systems for voltage-to-current transfer
        m_inv = np.linalg.inv(m)
        m_transformed_inv = np.linalg.inv(m_transformed)

        a = -m_inv @ r
        b = m_inv

        a_transformed = -m_transformed_inv @ r_transformed
        b_transformed = m_transformed_inv

        # Suppose we have V = 1.0 volt I = 1.0 amp in every coil,
        # and the structure filaments start at rest and remain unforced.
        # What does each system give us for dI/dt for the coils?
        u = np.ones(ncoils + nstructs)  # [V] input voltage
        x = np.ones(ncoils + nstructs)  # [A] initial current
        u[ncoils:] = 0.0
        x[ncoils:] = 0.0
        dIdt = a @ x + b @ u

        u_transformed = np.ones(ncoils + nmodes)  # [V] input voltage
        x_transformed = np.ones(ncoils + nmodes)  # [A] initial current
        u_transformed[ncoils:] = 0.0
        x_transformed[ncoils:] = 0.0
        dIdt_transformed = a_transformed @ x_transformed + b_transformed @ u_transformed

        # The divertor coils and VS coils will have a lot of error
        # for nearly any choice of number of modes to keep - even
        # keeping >500 modes only reduces their error by about
        # a factor of 2 compared to 40 modes.
        # NOTE: Is there a physical criterion we could use to group coils instead,
        #       like comparing self inductances and de-prioritizing low-inductance coils?
        # NOTE: Consider evaluating response to each coil being charged individually
        #       in addition to a simultaneous voltage applied to all coils
        # To handle this, we can keep a looser tolerance for
        # the DIV and VS coils, sacrificing accuracy in their
        # results for the ability to simulate the rest of the system.
        atol = 25.0  # [A/s] allow a dead zone for coils with near zero response
        rtol_pf_cs = 0.001
        #   DV and VS coils are more sensitive, but should still match well since we're not truncating modes
        rtol_div_vs = 0.01
        inds_pf_cs = [
            i
            for i, c in enumerate(coils)
            if any([x in c.name.lower() for x in ["pf", "cs"]])
        ]
        inds_div_vs = [
            i
            for i, c in enumerate(coils)
            if any([x in c.name.lower() for x in ["dv", "div", "vs"]])
        ]
        inds_all = inds_pf_cs + inds_div_vs
        duplicates_exist = len(list(set(inds_all))) != len(inds_all)
        coils_missed = len(inds_all) != ncoils
        if duplicates_exist:
            # Coils shouldn't appear in both lists, but there's nothing specifically
            # preventing this from happening. If it does happen, we need to use
            # better filters.
            raise NotImplementedError(
                "Filtered coil groups need update due to duplicates"
            )
        if coils_missed:
            # Similarly, if there were coils that didn't get captured by either filter,
            # then we need to account for that category.
            raise NotImplementedError(
                "Filtered coil groups need update due to missed coils"
            )

        dIdt_pf_cs = dIdt[inds_pf_cs]
        dIdt_div_vs = dIdt[inds_div_vs]

        dIdt_transformed_pf_cs = dIdt_transformed[inds_pf_cs]
        dIdt_transformed_div_vs = dIdt_transformed[inds_div_vs]

        # We have to do a two-sided comparison because th error may be large enough
        # that the assumption that the error is small compared to the nominal value
        # does not necessarily hold.
        for i in range(len(dIdt_pf_cs)):
            assert dIdt_pf_cs[i] == approx(
                dIdt_transformed_pf_cs[i], rel=rtol_pf_cs, abs=atol
            )
        for i in range(len(dIdt_div_vs)):
            assert dIdt_div_vs[i] == approx(
                dIdt_transformed_div_vs[i], rel=rtol_div_vs, abs=atol
            )

        plt.sca(axes[iax])
        plt.scatter(
            range(ncoils),
            dIdt[:ncoils],
            color="k",
            marker="x",
            alpha=0.4,
            s=25,
            # linestyle="--",
            label="Full System",
        )
        plt.scatter(
            range(ncoils),
            dIdt_transformed[:ncoils],
            color="k",
            s=10,
            # linestyle="-",
            label="Reduced System",
        )
        plt.gca().set_yscale("log")
        plt.gca().set_xticks(range(ncoils))
        plt.gca().set_xticklabels([x.name for x in device.coils], rotation=90.0)
        plt.xlabel("Coil Number")
        plt.ylabel("dI/dt [A/s]")
        plt.title(f"{device.model_reduction_method}\n{device._n_radial_slices} slices")
        plt.legend()

    plt.show()
