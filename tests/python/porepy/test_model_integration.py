"""Integration tests of the mpxa C++ code into PorePy models. Currently uses the
SinglePhaseFlow model. Makes a few time steps with the baseline (Python) and tested
(C++) discretizations and compares that results are the same within a tolerance."""

from copy import deepcopy

import numpy as np
import porepy as pp
import pytest
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)

import mpxa
# from mpxa.porepy_ad_bridge import ConcatenationOperator


class TestProblemBC(pp.SinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.west] = self.units.convert_units(5, "Pa")
        values[domain_sides.east] = self.units.convert_units(2, "Pa")
        return values


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("grid_type", ["cartesian", "simplex"])
@pytest.mark.parametrize("discr_type", ["tpfa", "mpfa"])
def test_fluid_mass_balance_model_integration(
    grid_type: str, discr_type: str, dim: int
):
    """Run the same single phase flow simulation with Python and C++ discretizations and
    ensure results are within the tolerance. The model is has 4 matrix cells in 2D and 8
    cells in 3D for cartesian grids, and reasonably more cells for a simplex grid. The
    model also includes intersecting fractures.

    """
    # Choosing the 2D or 3D geometry mixin.
    if dim == 2:
        GeometryMixin = SquareDomainOrthogonalFractures
    elif dim == 3:
        GeometryMixin = CubeDomainOrthogonalFractures
    else:
        raise ValueError

    # Creating the tested (C++) and expected (Python) models.
    class TestedModel(GeometryMixin, TestProblemBC):
        def darcy_flux_discretization(
            self, subdomains: list[pp.Grid]
        ) -> pp.ad.Discretization:
            if discr_type == "mpfa":
                return mpxa.MpfaAd(self.darcy_keyword, subdomains)
            elif discr_type == "tpfa":
                return mpxa.TpfaAd(self.darcy_keyword, subdomains)
            else:
                raise ValueError(f"{discr_type = }")

    class ExpectedModel(GeometryMixin, TestProblemBC):
        def darcy_flux_discretization(
            self, subdomains: list[pp.Grid]
        ) -> pp.ad.Discretization:
            if discr_type == "mpfa":
                return pp.ad.MpfaAd(self.darcy_keyword, subdomains)
            elif discr_type == "tpfa":
                return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
            else:
                raise ValueError(f"{discr_type = }")

    # Creating identical parameters for the tested and expected models.
    model_params_tested = {
        "grid_type": grid_type,
        "meshing_arguments": {"cell_size": 0.5},
        "time_manager": pp.TimeManager(schedule=[0, 5], dt_init=1, constant_dt=True),
        "fracture_indices": [0, 1] if dim == 2 else [0, 1, 2],  # Enabling intersecting fractures.
        "material_constants": {
            "fluid": pp.FluidComponent(viscosity=0.1, density=0.2),
            "solid": pp.SolidConstants(permeability=0.5, porosity=0.25),
        },
    }
    model_params_expected = deepcopy(model_params_tested, memo={})
    tested_model = TestedModel(model_params_tested)
    expected_model = ExpectedModel(model_params_expected)

    solver_params = {}
    pp.run_time_dependent_model(tested_model, solver_params)
    pp.run_time_dependent_model(expected_model, solver_params)

    # Fetching the primary fields (pressure and interface_darcy_flux) and comparing
    # their values.
    pressure_tested = tested_model.equation_system.evaluate(
        tested_model.pressure(tested_model.mdg.subdomains())
    )
    pressure_expected = expected_model.equation_system.evaluate(
        expected_model.pressure(expected_model.mdg.subdomains())
    )
    intf_flow_tested = tested_model.equation_system.evaluate(
        tested_model.interface_darcy_flux(tested_model.mdg.interfaces())
    )
    intf_flow_expected = expected_model.equation_system.evaluate(
        expected_model.interface_darcy_flux(expected_model.mdg.interfaces())
    )

    np.testing.assert_allclose(
        pressure_tested, pressure_expected, atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        intf_flow_tested, intf_flow_expected, atol=1e-14, rtol=1e-14
    )
