#ifndef MULTIPOINT_COMMON_H
#define MULTIPOINT_COMMON_H

#include <array>
#include <map>
#include <vector>

#include "grid.h"

// This file contains common definitions and includes for the multipoint flux and
// stress methods.

class BasisConstructor
{
   public:
    BasisConstructor(const int dim);
    ~BasisConstructor() = default;

    // Compute gradients of the linear basis functions for the simplex defined by coords.
    // Returns one gradient vector per vertex; each gradient has m_dim components.
    std::vector<std::array<double, 3>> compute_basis_functions(
        const std::vector<std::array<double, 3>>& coords);

   private:
    int m_dim;

    std::vector<std::array<double, 3>> compute_basis_functions_2d(
        const std::vector<std::array<double, 3>>& coords);
    std::vector<std::array<double, 3>> compute_basis_functions_3d(
        const std::vector<std::array<double, 3>>& coords);
};

class InteractionRegion
{
   public:
    // Constructor
    InteractionRegion(const int node, const int dim, const Grid& grid);

    // Destructor
    ~InteractionRegion() = default;

    // Getters for the interaction region data.
    // Now returns a map from face index to running index
    const std::map<int, int>& faces() const
    {
        return m_faces;
    }
    const std::vector<int>& cells() const
    {
        return m_cells;
    }
    const std::map<int, std::vector<int>>& faces_of_cells() const
    {
        return m_faces_of_cells;
    }
    const std::vector<int>& main_cell_of_faces() const
    {
        return m_main_cell_of_faces;
    }

   private:
    // Map from face index to running index (starting from 0)
    std::map<int, int> m_faces;
    std::vector<int> m_cells;
    // For each cell its associated faces. There will be nd of these.
    std::map<int, std::vector<int>> m_faces_of_cells;
    // For each face a single cell used to compute the flux.
    std::vector<int> m_main_cell_of_faces;
    // The node around which the interaction region is built.
    const int m_node;
    // The dimension of the problem to be discretize (do not confuse with the dimension
    // of the grid). How will this impact the topology of the interaction region?
    // We may decide to delete this, decide when we get to the vector problem.
    const int m_dim;
};

#endif  // MULTIPOINT_COMMON_H