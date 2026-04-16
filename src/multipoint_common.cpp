#include "../include/multipoint_common.h"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

// region BasisConstructor

BasisConstructor::BasisConstructor(const int dim) : m_dim(dim) {}

std::vector<std::array<double, 3>> BasisConstructor::compute_basis_functions(
    const std::vector<std::array<double, 3>>& coords)
{
    if (m_dim == 2)
    {
        return compute_basis_functions_2d(coords);
    }
    else if (m_dim == 3)
    {
        return compute_basis_functions_3d(coords);
    }
    throw std::runtime_error("MPFA basis function computation not implemented for dim != 2, 3.");
}

std::vector<std::array<double, 3>> BasisConstructor::compute_basis_functions_2d(
    const std::vector<std::array<double, 3>>& coords)
{
    double x_avg = 0.0, y_avg = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        x_avg += coords[i][0];
        y_avg += coords[i][1];
        if (std::abs(coords[i][2]) > 1e-20)
        {
            throw std::logic_error("Assumed z coordinate equals zero.");
        }
    }
    x_avg /= 3.0;
    y_avg /= 3.0;

    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        sx = std::max(sx, std::abs(coords[i][0] - x_avg));
        sy = std::max(sy, std::abs(coords[i][1] - y_avg));
    }
    if (sx == 0.0) sx = 1.0;
    if (sy == 0.0) sy = 1.0;

    const double inv_sx = 1.0 / sx;
    const double inv_sy = 1.0 / sy;

    constexpr double a11 = 1.0, a21 = 1.0, a31 = 1.0;
    const double a12 = (coords[0][0] - x_avg) * inv_sx, a13 = (coords[0][1] - y_avg) * inv_sy;
    const double a22 = (coords[1][0] - x_avg) * inv_sx, a23 = (coords[1][1] - y_avg) * inv_sy;
    const double a32 = (coords[2][0] - x_avg) * inv_sx, a33 = (coords[2][1] - y_avg) * inv_sy;

    const double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
                       a13 * (a21 * a32 - a22 * a31);
    if (std::abs(det) < 1e-20)
    {
        throw std::logic_error("compute_basis_functions det = 0.");
    }
    const double inv_det = 1.0 / det;

    // Gradient rows of the inverse (rows 1 and 2 correspond to x and y partials).
    double inv[3][3];
    inv[1][0] = (a23 * a31 - a21 * a33) * inv_det * inv_sx;
    inv[1][1] = (a11 * a33 - a13 * a31) * inv_det * inv_sx;
    inv[1][2] = (a13 * a21 - a11 * a23) * inv_det * inv_sx;
    inv[2][0] = (a21 * a32 - a22 * a31) * inv_det * inv_sy;
    inv[2][1] = (a12 * a31 - a11 * a32) * inv_det * inv_sy;
    inv[2][2] = (a11 * a22 - a12 * a21) * inv_det * inv_sy;

    std::vector<std::array<double, 3>> basis_functions(3);
    for (int j = 0; j < 3; ++j)
    {
        basis_functions[j][0] = inv[1][j];
        basis_functions[j][1] = inv[2][j];
        basis_functions[j][2] = 0.0;
    }
    return basis_functions;
}

std::vector<std::array<double, 3>> BasisConstructor::compute_basis_functions_3d(
    const std::vector<std::array<double, 3>>& coords)
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
    if (sx == 0.0) sx = 1.0;
    if (sy == 0.0) sy = 1.0;
    if (sz == 0.0) sz = 1.0;

    const double inv_sx = 1.0 / sx;
    const double inv_sy = 1.0 / sy;
    const double inv_sz = 1.0 / sz;

    constexpr double a11 = 1.0, a21 = 1.0, a31 = 1.0, a41 = 1.0;
    const double a12 = (coords[0][0] - x_avg) * inv_sx, a13 = (coords[0][1] - y_avg) * inv_sy,
                 a14 = (coords[0][2] - z_avg) * inv_sz;
    const double a22 = (coords[1][0] - x_avg) * inv_sx, a23 = (coords[1][1] - y_avg) * inv_sy,
                 a24 = (coords[1][2] - z_avg) * inv_sz;
    const double a32 = (coords[2][0] - x_avg) * inv_sx, a33 = (coords[2][1] - y_avg) * inv_sy,
                 a34 = (coords[2][2] - z_avg) * inv_sz;
    const double a42 = (coords[3][0] - x_avg) * inv_sx, a43 = (coords[3][1] - y_avg) * inv_sy,
                 a44 = (coords[3][2] - z_avg) * inv_sz;

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

    // Gradient rows of the inverse (rows 1-3 correspond to x, y, z partials).
    double inv[4][4];
    for (int j = 0; j < 4; ++j)
    {
        inv[1][j] = adj[j][1] * inv_detA * inv_sx;
        inv[2][j] = adj[j][2] * inv_detA * inv_sy;
        inv[3][j] = adj[j][3] * inv_detA * inv_sz;
    }

    std::vector<std::array<double, 3>> basis_functions(4);
    for (int j = 0; j < 4; ++j)
    {
        basis_functions[j][0] = inv[1][j];
        basis_functions[j][1] = inv[2][j];
        basis_functions[j][2] = inv[3][j];
    }
    return basis_functions;
}

// endregion BasisConstructor

// region InteractionRegion

InteractionRegion::InteractionRegion(const int node, const int dim, const Grid& grid)
    : m_node(node), m_dim(dim), m_faces(), m_cells(), m_faces_of_cells(), m_main_cell_of_faces()
{
    // Initialize the interaction region based on the node and the grid.
    auto face_indices_span = grid.faces_of_node(node);
    // Copying the face_indices data to a new vector to sort it.
    std::vector<int> face_indices(face_indices_span.begin(), face_indices_span.end());
    // Sort the face indices to ensure consistent ordering.
    std::sort(face_indices.begin(), face_indices.end());

    // Fill m_faces as a map from face index to running index
    for (size_t i = 0; i < face_indices.size(); ++i)
    {
        m_faces[face_indices[i]] = static_cast<int>(i);
    }

    m_cells.reserve(face_indices.size());
    m_main_cell_of_faces = std::vector<int>(face_indices.size(), -1);

    // For the cell indexing, use an unordered set for fast lookup, store the cells
    // in a vector to maintain the order of insertion.
    std::unordered_set<int> cell_set(m_cells.begin(), m_cells.end());

    for (size_t i = 0; i < face_indices.size(); ++i)
    {
        const int face = face_indices[i];
        // Get the cells associated with the face.
        const auto cells_of_face = grid.cells_of_face(face);
        // Associate the first cell with the face for the flux computation. There will
        // always be at least one cell associated with the face.
        m_main_cell_of_faces[i] = cells_of_face[0];

        // Loop over the cells to do two things:
        // 1. Add the cell to m_cells, thereby assign it a local index, but only if it
        //    is not already present.
        // 2. Assign this face to the set of faces of the cell.

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
                m_faces_of_cells[cell].reserve(m_dim);
            }
            m_faces_of_cells[cell].push_back(face);
        }
    }
    m_cells.resize(cell_set.size());
    // Sort the cells for consistency in the output.
    std::sort(m_cells.begin(), m_cells.end());
}
// endregion InteractionRegion