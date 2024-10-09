"""
Eigenmode-based dimensionality reduction for the part of the inductor
system that is a coupled L-R system - that is, conducting structure and
and coils that are shorted.
Shorted coils are taken as a kind of conducting structure here.

Suppose we have a symmetric NxN system mutual inductance matrix like this:
```
    Mutual Inductance
    Coils      Structure
 __________ _______________
|          |               |
|     0    |       2       | P
|          |               |
|__________|_______________|
|          |               |
|          |               |
|     3    |       1       | Q
|          |               |
|          |               |
|__________|_______________|
      P            Q
```

The regions are

* 0:   Coil-coil interaction, PxP
* 1:   Structure-Structure interaction QxQ
* 2,3: Coil-structure interaction, PxQ, QxP, with M_2.t() == M_3

...where P + Q = N.

We need to shrink regions 1,2,3 because while there are typically only
around 10 coils in region 0, there may be hundreds or thousands of
conducting structure elements contributing to the dimension of 1,2,3,
resulting in O(1e4)-O(1e6) nonzeros.

We'd rather have around O(1e2)-O(1e3) nonzeros in the entire
mutual inductance matrix, and we can sacrifice some modeling resolution
w.r.t. the conducting structure in order to achieve that dimensionality
reduction while preserving the detail of the active coils in region 0.

In order to do this, we need a non-square system transformation matrix
of shape KxQ that can reduce the QxQ region 1 to some smaller size QxK,
and can also reduce the PxQ,QxP regions 2,3 to PxK,KxP while preserving their
effect the overall system behavior w.r.t. the coil region 0 as well
as possible.

Eigenvector decomposition is a good candidate for this - the matrix
of eigenvectors preserves the behavior of the system. It also gives
a convenient ordering of the importance of each row and column
(the eigenvalues) in order to allow us to truncate the matrix
of eigenvectors and arrive at a truncated KxQ shape.

Now, it's not specifically the mutual inductance matrix itself
that we need to shrink, but the whole corresponding state-space
system. In this case, regions 1,2,3 (passive structure) relate to a
coupled inductive-resistive system, with self-interaction time scales
```
    A = -inv(M_1) @ R,                 # units of (1/s)
```
where
M_1 is the portion of M in region 1, and
R is the diagonal matrix of loop resistances associated with each
piece of conducting structure or shorted coil:

```
Structure
resistances
 _______________
| .             |
|    .          |
|       R       |  Q
|          .    |
|             . |
|_______________|
        Q
```

So, we can get the transformation matrix we need by taking the
top few eigenvectors of A:
```
    Tuv = V_sorted[:, :K]         # dimensionless, sampling K columns
```
where K is an empirically-tuned integer value with, hopefully, K << Q.

This gives a rectangular transformation matrix that fits our need:
```
 __________
|          |
|          |
|    Tuv   | Q
|          |
|          |
|__________|
      K
```
and can be used as an adapter between
the reduced-order model of the structure self-interaction represented by
```
    M1*  = Tuv' @ (M1 @ Tuv)       # Units of H, reduced-order structure-structure model
```
and the coupling regions 2,3
```
    M_2* = M_2 @ Tuv     # Units of H
    M_3* = M_3 @ Tuv
```
as well as tensors tabulating linear field contributions from
each structural element (flux, B-field, mutual inductance, etc):
```
    psi* = psi @ Tuv     # With psi tables shaped like (W, Q) for W grid points
    br*  = br  @ Tuv
    ...and so on.
```
Notably, the transformation of the resistances needs different treatment
because it's really a column vector that's been stretched into a diagonal matrix:
```
    R*   = Tuv' @ (R @ Tuv)    # Expands vector information in a similar way to outer(R, R)
```
As a final note, while the system for dI/dt uses the inverse of M, we really want the smallest
eigenvalues that would come out of that - the lowest frequency modes, which are most relevant
to controls. To avoid taking the (very ill-conditioned) inverse of M, we can instead solve the
system for `1/dI/dt = (-R^-1 @ M) @ v` and take the largest-magnitude eigenvalues with significantly
reduced numerical error.

This is not the _only_ reasonable choice for formulating the transformation matrix.
In particular, some other methods offer particular advantages in exchange for complexity:
    * Iterative SVD can be faster than proper eigenvalue decomposition
    * State-space balanced reduction gives a closer approximation of the important
      parts of system dynamics, especially phase response, but can suffer numerically with large systems
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def eigenmode_reduction(
    m: NDArray, r: NDArray, max_neig: int | None
) -> Tuple[NDArray, NDArray, int]:
    """
    Do eigenmode decomposition of inductive-resistive system and truncate
    to top `max_neig` terms.

    See module-level docs for more detail about the model reduction approach.

    Args:
        m: [H] QxQ symmetric mutual inductance matrix for conducting structure
        r: [ohm]     QxQ diagonal resistance matrix for conducting structure
        max_neig: Optional maximum number of eigenvalues to keep. None -> keep all.

    Returns:
        (d, tuv, neig), Eigenvalues in [s], QxK transformation matrix with K <= Q, and number of terms
    """
    rvec = np.diag(r)
    r_inv = np.diag(1.0 / rvec)  # [ohm^-1]

    # Build system describing 1/d(current)/d(time) timescales
    # under zero applied voltage (balanced inductive-resistive decay)
    # with unit current.
    a = -r_inv @ m  # [s] Coupled inductive-resistive system timescales

    # Do eigendecomposition using method for asymmetric matrices.
    # Per tradition, we have eigenvalues `d` and eigenvectors `v`.
    d, v = np.linalg.eig(a)  # ([s], [dimensionless])

    # Sort eigenvalues by magnitude and keep the permutation,
    # then reorder to lead with largest terms
    inds = np.flip(np.argsort(d**2))  # Sort for largest magnitude eigenvalues first
    tuv = v[:, inds]
    d = d[inds]

    # Truncate to take just the highest-magnitude terms
    if max_neig is not None:
        tuv = v[:, :max_neig]
        d = d[:max_neig]

    # The actual number of terms retained, which may be less
    # than the maximum allowable if the size of the supplied
    # system was smaller than max_neig.
    neig = len(d)

    return d, tuv, neig


def stabilized_eigenmode_reduction(
    m: NDArray, r: NDArray, max_neig: int | None
) -> tuple[NDArray, NDArray, int]:
    """
    This method can be more computationally stable than the direct method,
    and may give better results for slightly-asymmetric systems that are
    intended to be fully physically symmetric.

    This is essentially direct (eigendecomposition) PCA.

    See module-level docs for more detail about the model reduction approach.

    Args:
        m: [H] QxQ symmetric mutual inductance matrix for conducting structure
        r: [ohm] QxQ diagonal resistance matrix for conducting structure
        max_neig: Optional maximum number of eigenvalues to keep. None -> keep all.

    Returns:
        (d, tuv, neig), Eigenvalues in [s], QxK transformation matrix with K <= Q, and number of terms
    """
    # Unpack
    rvec = np.diag(r)  # [ohm] Extract diagonal
    r_inv = np.diag(1.0 / rvec)  # [ohm^-1]

    # Build system describing 1/(d(current)/d(time)) timescales
    # Then, take the covariance matrix to set up for direct PCA.
    #
    # To get a result with the correct units here, we have the option
    # to either take the elementwise square root now, or take the elementwise
    # square root of the solved eigenvalues. The correct choice is to take
    # the square root of the solved eigenvalues, because applying a nonlinear
    # function to the covariance matrix gives unfortunate effects including
    # nonphysical negative eigenvalues, and does not agree with the linear
    # algebra logic of PCA.
    a = -r_inv @ m  # [s] same system as the other method
    cov = a.T @ a  # [s^2] covariance matrix, symmetric positive semidefinite

    # Do eigendecomposition using method for Hermitian matrices
    # because this one is conveniently real and symmetric,
    # and this improves speed and numerical error.
    # Per tradition, we have eigenvalues `d` and eigenvectors `v`.
    d, v = np.linalg.eigh(cov)  # ([s^2], [dimensionless])
    #    Get the actual timescales instead of the square
    #    These corrected eigenvalues are the "loadings" in PCA jargon
    assert np.all(d >= 0.0), "Eigenvalues of a covariance matrix should be positive"
    d = np.sqrt(d)  # [s]

    # Sort eigenvalues by magnitude and keep the permutation,
    # then reorder to lead with largest terms
    inds = np.flip(np.argsort(d))
    d = d[inds]
    tuv = v[:, inds]

    # Truncate to take just the highest-magnitude terms
    if max_neig is not None:
        tuv = tuv[:, :max_neig]
        d = d[:max_neig]

    # The actual number of terms retained, which may be less
    # than the maximum allowable if the size of the supplied
    # system was smaller than max_neig.
    neig = len(d)

    # Make sure outputs are contiguous in memory to speed up access later
    d = np.ascontiguousarray(d)
    tuv = np.ascontiguousarray(tuv)

    return d, tuv, neig
