"""A manufacturer's published physical figure, as a contract constant (spec §3, §5).

Each contract module (`ros_surfaces/myagv.py`, `ros_surfaces/so101.py`,
`ros_surfaces/ainex/topics.py`) lists its robot's figures as `PHYSICAL_FIGURES`: what the
manufacturer publishes about the physical robot -- dimensions, mass, degrees of freedom,
speeds, servo figures -- each with the page it is published on, the page's own wording, and
the tolerance the compiled model is held to. `shared/tests/physical_figures_check.py`
measures every one on the compiled model and fails on any outside its tolerance.

A figure's tolerance is part of the contract, not a fudge: each one says why it is what it
is (`basis`), so widening one is a reviewed change like any other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Figure:
    """One published figure. `key` names the measurement the check makes."""

    key: str
    label: str
    value: float
    unit: str
    tolerance: float  # absolute, in `unit`; the model passes when |model - value| <= this
    source: str  # the page it is published on
    quote: str  # the page's own wording
    basis: str = ""  # why the tolerance is what it is, and how the figure is measured

    def within(self, measured: float) -> bool:
        return abs(float(measured) - self.value) <= self.tolerance + 1e-12


def percent(value: float, pct: float) -> float:
    """An absolute tolerance of `pct` percent of `value`."""
    return abs(value) * pct / 100.0


#: The tolerance on a dimension read off a vendor's visual meshes. The meshes are the
#: geometry the simulator may not change; the published size is a rounded measurement of
#: an assembled kit (cables, screw heads, printed tolerances) that the CAD omits.
MESH_DIMENSION_PCT = 5.0
