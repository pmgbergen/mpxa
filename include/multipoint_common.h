#ifndef MULTIPOINT_COMMON_H
#define MULTIPOINT_COMMON_H

#include <Eigen/Dense>
#include <array>
#include <map>
#include <vector>

#include "grid.h"

using Eigen::MatrixXd;

// This file contains common definitions and includes for the multipoint flux and
// stress methods.

class BasisConstructor
{
   public:
    // Constructor
    BasisConstructor(const int dim);

    // Destructor
    ~BasisConstructor() = default;

    // Function to compute the basis functions and their gradients. The signature is
    // highly uncertain.
    std::vector<std::array<double, 3>> compute_basis_functions(
        const std::vector<std::array<double, 3>>& coords);

   private:
    // Dimension of the problem
    int m_dim;
    // Matrix to store the coordinates of the nodes. Will be the left-hand side of the
    // equation.
    MatrixXd m_coord_matrix;
    // Matrix to store the computed basis functions.
    MatrixXd m_basis_matrix;
    // Matrix to represent the right hand side. Should be an identity matrix.
    MatrixXd m_rhs_matrix;
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