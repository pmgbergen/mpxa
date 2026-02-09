"""Test script to investigate model integration in porepy.

Observations:
- "python model_integration.py" runs with error due to inconsistency of linear algebra (dimensions).
- When changing code in mpxa, need to reinstall (-e does not work?). Use:
> pip install .[testing] && pip install . && python tests/python/porepy/model_integration.py
- Use MpxaFlowDiscretization to switch between Tpfa and Mpfa.

"""

import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
import mpxa 

class MpxaFlowDiscretization:

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]):
        return mpxa.TpfaAd(self.darcy_keyword, subdomains)
        #return mpxa.MpfaAd(self.darcy_keyword, subdomains)


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.units.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [] #frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method to assign simplex instead.

        """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        cell_size = self.units.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

class ModifiedBC(BoundaryConditionsSinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        # See section on scaling for explanation of the conversion.
        values[domain_sides.west] = self.units.convert_units(5, "Pa")
        values[domain_sides.east] = self.units.convert_units(2, "Pa")
        return values

class SinglePhaseFlowGeometryBC(
    MpxaFlowDiscretization,
    ModifiedGeometry,
    ModifiedBC,
    SinglePhaseFlow):
    """Adding both geometry and modified boundary conditions to the default model."""
    ...

fluid_constants = pp.FluidComponent(viscosity=0.1, density=0.2)
solid_constants = pp.SolidConstants(permeability=0.5, porosity=0.25)
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
model_params = {"material_constants": material_constants}

model = SinglePhaseFlowGeometryBC() #model_params)
pp.run_time_dependent_model(model)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), linewidth=0.25, title="Pressure distribution", plot_2d=True)
