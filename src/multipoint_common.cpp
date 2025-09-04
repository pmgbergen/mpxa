#include "../include/multipoint_common.h"

#include <cmath>
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

std::vector<std::array<double, 3>> BasisConstructor::compute_basis_functions(
    const std::vector<std::array<double, 3>>& coords)
{
    double inv[4][4];  // To store the inverse matrix.
    if (m_dim == 2)
    {
        double x_avg = 0.0, y_avg = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            x_avg += coords[i][0];
            y_avg += coords[i][1];
        }
        x_avg /= 3.0;
        y_avg /= 3.0;

        double sx = 0.0, sy = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            sx = std::max(sx, std::abs(coords[i][0] - x_avg));
            sy = std::max(sy, std::abs(coords[i][1] - y_avg));
        }

        // Avoid division by zero (degenerate triangle)
        if (sx == 0.0) sx = 1.0;
        if (sy == 0.0) sy = 1.0;

        const double inv_sx = 1.0 / sx;
        const double inv_sy = 1.0 / sy;

        // Extract elements of the matrix.
        constexpr double a11 = 1.0, a21 = 1.0, a31 = 1.0;
        // Scale the coordinates to improve numerical stability.
        double a12 = (coords[0][0] - x_avg) * inv_sx, a13 = (coords[0][1] - y_avg) * inv_sy;
        double a22 = (coords[1][0] - x_avg) * inv_sx, a23 = (coords[1][1] - y_avg) * inv_sy;
        double a32 = (coords[2][0] - x_avg) * inv_sx, a33 = (coords[2][1] - y_avg) * inv_sy;

        // Compute determinant
        double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
                     a13 * (a21 * a32 - a22 * a31);
        const double inv_det = 1.0 / det;

        // Compute inverse (assuming det != 0)
        // double inv[3][3];
        // inv[0][0] = (a22 * a33 - a23 * a32) * inv_det;
        // inv[0][1] = (a13 * a32 - a12 * a33) * inv_det;
        // inv[0][2] = (a12 * a23 - a13 * a22) * inv_det;
        inv[1][0] = (a23 * a31 - a21 * a33) * inv_det * inv_sx;
        inv[1][1] = (a11 * a33 - a13 * a31) * inv_det * inv_sx;
        inv[1][2] = (a13 * a21 - a11 * a23) * inv_det * inv_sx;
        inv[2][0] = (a21 * a32 - a22 * a31) * inv_det * inv_sy;
        inv[2][1] = (a12 * a31 - a11 * a32) * inv_det * inv_sy;
        inv[2][2] = (a11 * a22 - a12 * a21) * inv_det * inv_sy;
    }
    else
    {
        double x_avg = 0.0, y_avg = 0.0, z_avg = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            x_avg += coords[i][0];
            y_avg += coords[i][1];
            z_avg += coords[i][2];
        }
        x_avg /= 4.0;
        y_avg /= 4.0;
        z_avg /= 4.0;

        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            sx = std::max(sx, std::abs(coords[i][0] - x_avg));
            sy = std::max(sy, std::abs(coords[i][1] - y_avg));
            sz = std::max(sz, std::abs(coords[i][2] - z_avg));
        }

        // Avoid division by zero (degenerate tetrahedron)
        if (sx == 0.0) sx = 1.0;
        if (sy == 0.0) sy = 1.0;
        if (sz == 0.0) sz = 1.0;

        const double inv_sx = 1.0 / sx;
        const double inv_sy = 1.0 / sy;
        const double inv_sz = 1.0 / sz;

        constexpr double a11 = 1.0, a21 = 1.0, a31 = 1.0, a41 = 1.0;
        double a12 = (coords[0][0] - x_avg) * inv_sx, a13 = (coords[0][1] - y_avg) * inv_sy,
               a14 = (coords[0][2] - z_avg) * inv_sz;
        double a22 = (coords[1][0] - x_avg) * inv_sx, a23 = (coords[1][1] - y_avg) * inv_sy,
               a24 = (coords[1][2] - z_avg) * inv_sz;
        double a32 = (coords[2][0] - x_avg) * inv_sx, a33 = (coords[2][1] - y_avg) * inv_sy,
               a34 = (coords[2][2] - z_avg) * inv_sz;
        double a42 = (coords[3][0] - x_avg) * inv_sx, a43 = (coords[3][1] - y_avg) * inv_sy,
               a44 = (coords[3][2] - z_avg) * inv_sz;

        // Compute 3x3 minors for determinant
        auto det3x3 = [](double b11, double b12, double b13, double b21, double b22, double b23,
                         double b31, double b32, double b33)
        {
            return b11 * (b22 * b33 - b23 * b32) - b12 * (b21 * b33 - b23 * b31) +
                   b13 * (b21 * b32 - b22 * b31);
        };

        const double detA = a11 * det3x3(a22, a23, a24, a32, a33, a34, a42, a43, a44) -
                            a12 * det3x3(a21, a23, a24, a31, a33, a34, a41, a43, a44) +
                            a13 * det3x3(a21, a22, a24, a31, a32, a34, a41, a42, a44) -
                            a14 * det3x3(a21, a22, a23, a31, a32, a33, a41, a42, a43);
        const double inv_detA = 1.0 / detA;

        // Compute adjugate matrix (transpose of cofactor matrix)
        double adj[4][4];
        adj[0][0] = det3x3(a22, a23, a24, a32, a33, a34, a42, a43, a44);
        adj[0][1] = -det3x3(a21, a23, a24, a31, a33, a34, a41, a43, a44);
        adj[0][2] = det3x3(a21, a22, a24, a31, a32, a34, a41, a42, a44);
        adj[0][3] = -det3x3(a21, a22, a23, a31, a32, a33, a41, a42, a43);

        adj[1][0] = -det3x3(a12, a13, a14, a32, a33, a34, a42, a43, a44);
        adj[1][1] = det3x3(a11, a13, a14, a31, a33, a34, a41, a43, a44);
        adj[1][2] = -det3x3(a11, a12, a14, a31, a32, a34, a41, a42, a44);
        adj[1][3] = det3x3(a11, a12, a13, a31, a32, a33, a41, a42, a43);

        adj[2][0] = det3x3(a12, a13, a14, a22, a23, a24, a42, a43, a44);
        adj[2][1] = -det3x3(a11, a13, a14, a21, a23, a24, a41, a43, a44);
        adj[2][2] = det3x3(a11, a12, a14, a21, a22, a24, a41, a42, a44);
        adj[2][3] = -det3x3(a11, a12, a13, a21, a22, a23, a41, a42, a43);

        adj[3][0] = -det3x3(a12, a13, a14, a22, a23, a24, a32, a33, a34);
        adj[3][1] = det3x3(a11, a13, a14, a21, a23, a24, a31, a33, a34);
        adj[3][2] = -det3x3(a11, a12, a14, a21, a22, a24, a31, a32, a34);
        adj[3][3] = det3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33);

        // Compute inverse by dividing adjugate by determinant
        // double inv[4][4];
        // for (int i = 0; i < 4; ++i)
        // {

        for (int j = 0; j < 4; ++j)
        {
            inv[1][j] = adj[j][1] * inv_detA * inv_sx;
            inv[2][j] = adj[j][2] * inv_detA * inv_sy;
            inv[3][j] = adj[j][3] * inv_detA * inv_sz;
            // inv[i][j] = adj[j][i] * inv_detA;
        }
        // }
    }

    // for (int i = 0; i <= m_dim; ++i)
    // {
    //     // m_coord_matrix(i, 0) = 1.0;  // Set the first column to ones for the constant term.
    //     // Fill the coord_matrix with the coordinates provided in the input.
    //     for (int j = 1; j <= m_dim; ++j)
    //     {
    //         m_coord_matrix(i, j) = coords[i][j - 1];
    //     }
    // }

    // // Solve the linear system with m_coord_matrix as the left-hand side and
    // // m_rhs_matrix as the right-hand side.
    // Eigen::PartialPivLU<Eigen::MatrixXd> lu_decomp(m_coord_matrix);
    // m_basis_matrix = lu_decomp.solve(m_rhs_matrix);

    // // Store the computed basis functions in the output vector.
    std::vector<std::array<double, 3>> basis_functions(m_dim + 1);

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
            basis_functions[j][i - 1] = inv[i][j];  // m_basis_matrix(i, j);
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
    // Sort the face indices to ensure consistent ordering.
    std::sort(face_indices.begin(), face_indices.end());

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

    // Sort the cells and faces for consistency in the output.
    std::sort(m_cells.begin(), m_cells.end());
}
// endregion InteractionRegion