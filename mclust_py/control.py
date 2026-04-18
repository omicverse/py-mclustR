"""EM control parameters — mirror of mclust::emControl()."""
from __future__ import annotations

from dataclasses import dataclass


_EPS = 2.220446049250313e-16  # .Machine$double.eps in R


@dataclass
class EMControl:
    """Equivalent of R's `emControl()`.

    Attributes mirror the R defaults exactly so EM trajectories and
    convergence behaviour match `me<Model>` from the CRAN package.
    """

    eps: float = _EPS
    tol: tuple[float, float] = (1e-5, 1e-5)
    itmax: tuple[int, int] = (10**8, 10**8)
    equal_pro: bool = False


def em_control(
    eps: float = _EPS,
    tol: float | tuple[float, float] = 1e-5,
    itmax: int | tuple[int, int] = 10**8,
    equal_pro: bool = False,
) -> EMControl:
    """Build an :class:`EMControl` matching R's `emControl()` semantics."""
    if not isinstance(tol, tuple):
        tol = (float(tol), float(tol))
    if not isinstance(itmax, tuple):
        itmax = (int(itmax), int(itmax))
    return EMControl(eps=float(eps), tol=tol, itmax=itmax, equal_pro=bool(equal_pro))
