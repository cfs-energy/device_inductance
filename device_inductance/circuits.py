from dataclasses import dataclass

import numpy as np
from omas import ODS


@dataclass(frozen=True)
class CoilSeriesCircuit:
    """
    A circuit with a single power supply and one or more coils
    connected in series.
    """

    name: str
    """Human-friendly name of the circuit"""

    supply: int
    """Index of power supply for this circuit"""

    coils: list[tuple[int, float]]
    """
    Map of the coils that are in series on this circuit to their series orientation
    which should be either -1.0 or 1.0
    """

    def __post_init__(self):
        # Make sure the sign info is correctly formatted
        for i in range(len(self.coils)):
            c = self.coils[i]
            assert c[1] != 0.0, "Sign of zero will not conserve charge"
            self.coils[i] = (c[0], np.sign(c[1]))


def _extract_circuits(description: ODS) -> list[CoilSeriesCircuit]:
    # https://gafusion.github.io/omas/schema/schema_pf%20active.html
    # device_description takes the "sides" used in this schema as interleaved
    # so that the entries for each nodes are like [ps0 in, ps0 out, ...]
    # and we continue that convention here

    # Figure out how many power supplies there are
    # The connectivity info is stored as a sort of adjacency matrix,
    # with shape (nnodes, nsides) for each circuit where nsides = 2*(nsupplies + ncoils),
    # the number of inputs and outputs of circuit components.
    ncoils = len(description["pf_active.coil"].values())
    conn0_shape = description["pf_active.circuit.0"]["connections"].shape
    nsupplies = int(conn0_shape[1] / 2 - ncoils)
    ncomponents = ncoils + nsupplies
    assert (
        2 * (ncomponents) == conn0_shape[1]
    ), "Connection sides must have even dimension and match number of components"

    # For each circuit in the ODS, figure out what coils are connected
    # and in what orientation
    circuits: list[CoilSeriesCircuit] = []
    for ods_circuit in description["pf_active.circuit"].values():
        # Parse interleaved format
        name = ods_circuit["name"]
        connections = ods_circuit["connections"]
        in_connections = connections[:, :-1][:, ::2]  # Connections to "in" sides
        out_connections = connections[:, 1::2]  # Connections to "out" sides
        ps_out_connections = out_connections[:, :nsupplies]
        ps_in_connections = in_connections[:, :nsupplies]
        coil_out_connections = out_connections[:, nsupplies:]
        coil_in_connections = in_connections[:, nsupplies:]
        # First, make sure there are no parallel connections.
        # This means every "side" should be attached to no more than
        # 1 node, so we can just check the maximum value of the sum of nodes
        # connected to sides.
        assert (
            max(np.sum(connections, axis=0)) < 2
        ), f"Circuit {ods_circuit} contains parallel paths"

        # We could still have parallel paths if there are two disjoint circuits
        # represented. To check that, we can make sure there is only one power supply
        # involved in the circuit
        assert (
            np.sum(in_connections[:, :nsupplies].flatten()) == 1
        ), f"Circuit {ods_circuit} uses multiple or zero power supplies"

        assert (
            np.sum(out_connections[:, :nsupplies].flatten()) == 1
        ), f"Circuit {ods_circuit} uses multiple or zero power supplies"

        # Now that we know there are no parallel paths, we're interested in
        # which things are connected in series, and in what order.
        #     Which power supply are we using, and what are our starting and ending nodes?
        #     SIGN CONVENTION: power supplies nominally flow current from "out" to "in",
        #     while coils nominally flow current from "in" to "out" per discussion with
        #     Christoph Hasse on slack
        starting_node, ps_index = np.argwhere(ps_in_connections == 1)[0]
        ending_node, _ = np.argwhere(ps_out_connections == 1)[0]
        coils = []

        # For each node, see what coil is attached to it
        # and which side is attached. There will only be one
        # coil
        def get_next_coil(node):
            # Get the next unvisited coil attached to a given node, and its sign orientation
            visited_coil_indices = [x[0] for x in coils]
            inc = [
                np.squeeze(x)
                for x in np.argwhere(coil_in_connections[node, :])
                if x not in visited_coil_indices
            ]
            outc = [
                np.squeeze(x)
                for x in np.argwhere(coil_out_connections[node, :])
                if x not in visited_coil_indices
            ]
            if len(inc) == 1 and len(outc) == 0:
                # If we are traversing in the direction of positive current and see
                # the coil input first, it's oriented positive
                return int(inc[0]), 1.0
            elif len(inc) == 0 and len(outc) == 1:
                # If we are traversing in the direction of positive current and see
                # the coil output first, it's oriented negative
                return int(outc[0]), -1.0
            else:
                raise ValueError(
                    f"Multiple or zero unvisited coils {inc} and {outc} at node {node} "
                    "imply parallel system or open circuit"
                )

        def get_next_node(coil, sign):
            if sign == 1.0:
                return np.argwhere(coil_out_connections[:, coil] == 1)[0][0]
            elif sign == -1.0:
                return np.argwhere(coil_in_connections[:, coil] == 1)[0][0]
            else:
                raise ValueError(f"Invalid sign {sign} for coil {coil}")

        node = starting_node
        i = 0
        while node != ending_node:
            assert i <= ncoils, f"Circuit for supply {ps_index} exceeds possible length"
            coil, sign = get_next_coil(node)
            coils.append((coil, sign))
            node = get_next_node(coil, sign)
            i += 1

        circuit = CoilSeriesCircuit(
            name=name, coils=coils, supply=ps_index
        )  # Initialize output
        circuits.append(circuit)

    # Make sure all coils are represented exactly once
    all_coils_in_circuits = sum(
        [[c[0] for c in circuit.coils] for circuit in circuits], start=[]
    )
    for i in range(ncoils):
        assert i in all_coils_in_circuits, f"Coil {i} missing from circuits"
    assert (
        len(list(set(all_coils_in_circuits))) == ncoils
    ), "Extra coils or missing coils in circuits"

    return circuits
