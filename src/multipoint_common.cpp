
#include "../include/multipoint_common.h"

#include <iostream>
#include <unordered_set>

// region BasisConstructor

BasisConstructor::BasisConstructor(const int dim)
    : m_dim(dim), m_coord_matrix(), m_basis_matrix(), m_rhs_matrix()
{
    // Initialize the matrices with appropriate sizes
    m_coord_matrix = MatrixXd::Zero(dim + 1, dim + 1);
    // The first column of the coord matrix will be ones.
    for (int i = 0; i <= dim; ++i)
    {
        m_coord_matrix(i, 0) = 1.0;
    }

    m_basis_matrix = MatrixXd::Zero(dim + 1, dim + 1);
    m_rhs_matrix = MatrixXd::Identity(dim + 1, dim + 1);
}

std::vector<std::vector<double>> BasisConstructor::compute_basis_functions(
    const std::vector<std::array<double, 3>>& coords)
{
    // Compute the basis functions and their gradients. The implementation is highly
    // uncertain and will depend on the specific problem.
    // This is a placeholder for the actual implementation.

    for (int i = 0; i <= m_dim; ++i)
    {
        m_coord_matrix(i, 0) = 1.0;  // Set the first column to ones for the constant term.
        // Fill the coord_matrix with the coordinates provided in the input.
        for (int j = 1; j <= m_dim; ++j)
        {
            m_coord_matrix(i, j) = coords[i][j - 1];
        }
    }

    // Solve the linear system with m_coord_matrix as the left-hand side and
    // m_rhs_matrix as the right-hand side.

    m_basis_matrix = m_coord_matrix.lu().solve(m_rhs_matrix);

    // Store the computed basis functions in the output vector.
    std::vector<std::vector<double>> basis_functions(m_dim + 1, std::vector<double>(m_dim));

    // Some index gymnastics here: The basis functions are stored column-wise in the
    // m_basis_matrix, but we want to return them row-wise, stored in the vector of
    // vectors. Use j as the column index, i as the row index in the matrix (for the
    // basis function they become respectively the basis function counter and the
    // coordinate index). Also, having a tight loop over the outer index for one of the
    // variables seem unavoidable, unless we switch to a less intuitive data structure
    // for the basis functions, which would be worse for readability.
    for (int j = 0; j <= m_dim; ++j)
    {
        // Start indexing from 1, since the first row of the basis functions is the
        // constant term, which we ignore.
        for (int i = 1; i <= m_dim; ++i)
        {
            // Store the computed basis functions in the output vector.
            basis_functions[j][i - 1] = m_basis_matrix(i, j);
        }
    }
    return basis_functions;
}

// endregion BasisConstructor

// region InteractionRegion

InteractionRegion::InteractionRegion(const int node, const int dim, const Grid& grid)
    : m_node(node), m_dim(dim), m_faces(), m_cells(), m_faces_of_cells(), m_main_cell_of_faces()
{
    // Initialize the interaction region based on the node and the grid.
    auto face_indices = grid.faces_of_node(node);

    // Fill m_faces as a map from face index to running index
    for (size_t i = 0; i < face_indices.size(); ++i)
    {
        m_faces[face_indices[i]] = static_cast<int>(i);
    }

    m_cells = std::vector<int>();
    m_main_cell_of_faces = std::vector<int>(face_indices.size(), -1);

    m_faces_of_cells = std::map<int, std::vector<int>>();

    for (size_t i = 0; i < face_indices.size(); ++i)
    {
        const int face = face_indices[i];
        // Get the cells associated with the face.
        auto cells_of_face = grid.cells_of_face(face);
        // Associate the first cell with the face for the flux computation. There will
        // always be at least one cell associated with the face.
        m_main_cell_of_faces[i] = cells_of_face[0];

        // Loop over the cells to do two things:
        // 1. Add the cell to m_cells, thereby assign it a local index, but only if it
        //    is not already present.
        // 2. Assign this face to the set of faces of the cell.

        // For the cell indexing, use an unordered set for fast lookup, store the cells
        // in a vector to maintain the order of insertion.
        std::unordered_set<int> cell_set(m_cells.begin(), m_cells.end());

        for (const int cell : cells_of_face)
        {
            if (cell_set.insert(cell).second)
            {  // true if inserted
                m_cells.push_back(cell);
            }
            // Assign the face to the cell. If the cell has not been seen before, it will
            // be added to the map with the face as the first element of the vector.
            if (m_faces_of_cells.find(cell) == m_faces_of_cells.end())
            {
                // We know (rather assume) that each cell has exactly dim faces meeting
                // at the node (this will break for pyramid cells, but if we encounter
                // those, we will have all sorts of problems).
                m_faces_of_cells[cell] = std::vector<int>();
            }
            m_faces_of_cells[cell].push_back(face);
        }
    }
}
// endregion InteractionRegion