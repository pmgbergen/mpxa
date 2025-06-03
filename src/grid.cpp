#include "../include/grid.h"

#include <cmath>
#include <iostream>
#include <vector>

Grid::Grid(const int dim, std::vector<std::vector<double>> nodes,
           std::shared_ptr<CompressedDataStorage<int>> cell_faces,
           std::shared_ptr<CompressedDataStorage<int>> face_nodes)
    : m_dim(dim),
      m_nodes(std::move(nodes)),
      m_cell_faces(std::move(cell_faces)),
      m_face_nodes(std::move(face_nodes))
{
    m_num_nodes = m_face_nodes->num_rows();
    m_num_faces = m_face_nodes->num_cols();
    m_num_cells = m_cell_faces->num_cols();
    m_cell_volumes.resize(m_num_cells);
    m_face_areas.resize(m_num_faces);
    m_face_normals.resize(m_num_faces, std::vector<double>(m_dim));
    m_face_centers.resize(m_num_faces, std::vector<double>(m_dim));
    m_cell_centers.resize(m_num_cells, std::vector<double>(m_dim));
}

const std::vector<int> Grid::boundary_faces() const
{
    std::vector<int> boundary_faces_list;

    for (int face = 0; face < m_num_faces; ++face)
    {
        if (m_cell_faces->cols_in_row(face).size() == 1)
        {
            boundary_faces_list.push_back(face);
        }
    }
    return boundary_faces_list;
}

const int Grid::dim() const
{
    return m_dim;
}

// Getters for topological data
const int Grid::num_nodes() const
{
    return m_num_nodes;
}
const int Grid::num_cells() const
{
    return m_num_cells;
}
const int Grid::num_faces() const
{
    return m_num_faces;
}

const std::vector<int> Grid::faces_of_node(const int node) const
{
    return m_face_nodes->cols_in_row(node);
}

const std::vector<int> Grid::nodes_of_face(const int face) const
{
    return m_face_nodes->rows_in_col(face);
}

const std::vector<int> Grid::cells_of_face(const int face) const
{
    return m_cell_faces->cols_in_row(face);
}

const std::vector<int> Grid::faces_of_cell(const int cell) const
{
    return m_cell_faces->rows_in_col(cell);
}

const int Grid::sign_of_face_cell(const int face, const int cell) const
{
    return m_cell_faces->value(face, cell);
}

const int Grid::num_nodes_of_face(const int face) const
{
    return m_face_nodes->rows_in_col(face).size();
}

// Getters for geometric data
const std::vector<std::vector<double>>& Grid::nodes() const
{
    return m_nodes;
}
const std::vector<std::vector<double>>& Grid::cell_centers() const
{
    return m_cell_centers;
}
const std::vector<double>& Grid::cell_volumes() const
{
    return m_cell_volumes;
}
const std::vector<double>& Grid::face_areas() const
{
    return m_face_areas;
}
const std::vector<std::vector<double>>& Grid::face_normals() const
{
    return m_face_normals;
}
const std::vector<std::vector<double>>& Grid::face_centers() const
{
    return m_face_centers;
}
// Getters for individual elements
const std::vector<double>& Grid::cell_center(int cell) const
{
    return m_cell_centers[cell];
}
const double& Grid::cell_volume(int cell) const
{
    return m_cell_volumes[cell];
}
const double& Grid::face_area(int face) const
{
    return m_face_areas[face];
}
const std::vector<double>& Grid::face_normal(int face) const
{
    return m_face_normals[face];
}
const std::vector<double>& Grid::face_center(int face) const
{
    return m_face_centers[face];
}
// Setters for the geometry data, in case these are computed externally.
void Grid::set_cell_volumes(const std::vector<double>& cell_volumes)
{
    m_cell_volumes = cell_volumes;
}
void Grid::set_face_areas(const std::vector<double>& face_areas)
{
    m_face_areas = face_areas;
}
void Grid::set_face_normals(const std::vector<std::vector<double>>& face_normals)
{
    m_face_normals = face_normals;
}
void Grid::set_face_centers(const std::vector<std::vector<double>>& face_centers)
{
    m_face_centers = face_centers;
}
void Grid::set_cell_centers(const std::vector<std::vector<double>>& cell_centers)
{
    m_cell_centers = cell_centers;
}

void Grid::compute_geometry()
{
    // Loop over all faces, get them_nodes of the face, compute the face center and area
    // from the node coordinates. The loop does not work very well with the storage format
    // for node-face relation (which is a mapping fromm_nodes to faces), but it will have
    // to do for now.
    for (int i{0}; i < num_faces(); ++i)
    {
        // Get them_nodes of the face
        std::vector<int> loc_nodes = nodes_of_face(i);
        // Get the number ofm_nodes of the face
        const int num_nodes = loc_nodes.size();

        // Compute the face center. Loop over the dimensions and them_nodes of the face
        // and compute the center as the average of the node coordinates.
        for (int j{0}; j < dim(); ++j)
        {
            m_face_centers[i][j] = 0.0;
            for (int k{0}; k < num_nodes; ++k)
            {
                m_face_centers[i][j] += m_nodes[loc_nodes[k]][j];
            }
            m_face_centers[i][j] /= num_nodes;
        }
        // Compute the face area
        if (m_dim == 2)
        {
            const double dx = m_nodes[loc_nodes[1]][0] - m_nodes[loc_nodes[0]][0];
            const double dy = m_nodes[loc_nodes[1]][1] - m_nodes[loc_nodes[0]][1];
            m_face_areas[i] = std::sqrt(dx * dx + dy * dy);

            m_face_normals[i][0] = dy;
            m_face_normals[i][1] = -dx;
        }
        else  // m_dim == 3
        {
            if (num_nodes == 3)
            {
                // The face is a triangle. We can compute the area using the cross product
                // of two vectors in the plane of the triangle.
                std::vector<double> v1(3), v2(3);
                for (int j{0}; j < m_dim; ++j)
                {
                    v1[j] = m_nodes[loc_nodes[1]][j] - m_nodes[loc_nodes[0]][j];
                    v2[j] = m_nodes[loc_nodes[2]][j] - m_nodes[loc_nodes[0]][j];
                }
                // Compute the face normal vector as the cross product of v1 and v2.
                std::vector<double> normal(3);
                normal[0] = (v1[1] * v2[2] - v1[2] * v2[1]);
                normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
                normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
                // Compute the area as the magnitude of the normal vector.
                m_face_areas[i] = 0.5 * std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                                  normal[2] * normal[2]);

                // Set the face normal length to the face area. We may change the
                // direction later.
                for (int j{0}; j < m_dim; ++j)
                {
                    m_face_normals[i][j] = normal[j] / m_face_areas[i];
                }
            }

            else  // num_nodes == 4
            {
                // NOTE: This is only valid for grids aligned with the axes.
                // This is a quadrilateral. We cannot trust that them_nodes are ordered
                // in a circular fashion. Therefore, compute the maximum and minimum
                // coordinates along each axis. For one of the dimension, the max and
                // min will be equal. For the other two, we can compute the area by a
                // Cartesian product.
                std::vector<double> min_coords(3), max_coords(3);
                for (int j{0}; j < m_dim; ++j)
                {
                    min_coords[j] = m_nodes[loc_nodes[0]][j];
                    max_coords[j] = m_nodes[loc_nodes[0]][j];
                }
                for (int j{1}; j < num_nodes; ++j)
                {
                    for (int k{0}; k < m_dim; ++k)
                    {
                        min_coords[k] = std::min(min_coords[k], m_nodes[loc_nodes[j]][k]);
                        max_coords[k] = std::max(max_coords[k], m_nodes[loc_nodes[j]][k]);
                    }
                }
                // Compute the area as the product of the differences in the two dimensions
                // where the max and min are different.
                for (int j{0}; j < m_dim; ++j)
                {
                    // Initialize the face normal to zero
                    m_face_normals[i][j] = 0.0;
                }
                for (int j{0}; j < m_dim; ++j)
                    if (min_coords[j] == max_coords[j])
                    {
                        const int k = (j + 1) % m_dim;
                        const int l = (j + 2) % m_dim;
                        m_face_areas[i] =
                            (max_coords[k] - min_coords[k]) * (max_coords[l] - min_coords[l]);
                        // Set the face normal to an area weighted normal in the
                        // dimension of the face (where min and max are equal). We may
                        // change the direction later.
                        m_face_normals[i][j] = m_face_areas[i];
                        break;
                    }
            }
        }
    }

    // Loop over all cells, obtain the faces of the cell, compute the cell center and
    // volume from face information.
    for (int i{0}; i < m_num_cells; ++i)
    {
        std::vector<int> loc_faces = faces_of_cell(i);
        const int num_faces = loc_faces.size();

        m_cell_centers[i] = std::vector<double>(m_dim);

        // Compute the cell center. Loop over the dimensions and the faces of the cell
        // and compute the center as the average of the face centers. This will not work
        // for general polyhedra (nor polygons?), but it should do for simplices and
        // Cartesian grids.
        for (int j{0}; j < m_dim; ++j)
        {
            m_cell_centers[i][j] = 0.0;
            for (int k{0}; k < num_faces; ++k)
            {
                m_cell_centers[i][j] += m_face_centers[loc_faces[k]][j];
            }
            // Take the mean value.
            m_cell_centers[i][j] /= num_faces;
        }

        // Compute the cell volume. Loop over the faces of the cell and compute the
        // volume as the sum of the face areas times the distance from the face center to
        // the cell center.
        m_cell_volumes[i] = 0.0;
        for (int j{0}; j < num_faces; ++j)
        {
            // Create a vector from the face center to the cell center
            std::vector<double> face_to_cell(m_dim);
            for (int k{0}; k < m_dim; ++k)
            {
                face_to_cell[k] = m_cell_centers[i][k] - m_face_centers[loc_faces[j]][k];
            }
            // Project onto the face normal to get the distance from the face center to
            // the cell center.
            double dist = 0.0;
            for (int k{0}; k < m_dim; ++k)
            {
                dist +=
                    face_to_cell[k] * m_face_normals[loc_faces[j]][k] / m_face_areas[loc_faces[j]];
            }
            // Add the volume contribution from the face.
            dist = std::abs(dist);
            m_cell_volumes[i] += m_face_areas[loc_faces[j]] * dist / m_dim;
        }
    }
    // Finally, loop over faces, check that the normal vectors point out of the cell
    // for which face_cell_sign is positive. If not, change the sign of the normal
    // vector.
    for (int i{0}; i < m_num_faces; ++i)
    {
        const std::vector<int> loc_cells = cells_of_face(i);

        // Create a vector from the face center to the cell center
        double dot_prod = 0.0;
        for (int j{0}; j < m_dim; ++j)
        {
            const double face_to_cell = m_face_centers[i][j] - m_cell_centers[loc_cells[0]][j];
            dot_prod += face_to_cell * m_face_normals[i][j];
        }

        if ((sign_of_face_cell(i, loc_cells[0]) < 0 && dot_prod > 0) ||
            (sign_of_face_cell(i, loc_cells[0]) > 0 && dot_prod < 0))
        {
            for (int j{0}; j < m_dim; ++j)
            {
                m_face_normals[i][j] *= -1;
            }
        }
    }
}

// Cartesian grid creation
std::unique_ptr<Grid> Grid::create_cartesian_grid(const int dim, const std::vector<int> num_cells,
                                                  const std::vector<double> lengths)
{
    // Dim should be 2 or 3
    if (dim < 2 || dim > 3)
    {
        throw std::invalid_argument("Invalid dimension: dim must be 2 or 3.");
    }

    // Create node coordinates along each dimension
    std::vector<double> x(num_cells[0] + 1);
    std::vector<double> y(num_cells[1] + 1);
    std::vector<double> z =
        dim == 3 ? std::vector<double>(num_cells[2] + 1) : std::vector<double>();

    double dx = lengths[0] / num_cells[0];
    double dy = lengths[1] / num_cells[1];
    double dz = dim == 3 ? lengths[2] / num_cells[2] : 0.0;

    for (int i = 0; i < num_cells[0] + 1; ++i)
    {
        x[i] = i * dx;
    }
    for (int i = 0; i < num_cells[1] + 1; ++i)
    {
        y[i] = i * dy;
    }

    if (dim == 3)
    {
        for (int i = 0; i < num_cells[2] + 1; ++i)
        {
            z[i] = i * dz;
        }
    }

    // Bookkeeping: First the total number of nodes..
    int num_nodes = (num_cells[0] + 1) * (num_cells[1] + 1);
    if (dim == 3)
    {
        num_nodes *= (num_cells[2] + 1);
    }
    // ..then the number of nodes along each dimension.
    std::vector<int> num_nodes_per_dim(3);
    num_nodes_per_dim[0] = num_cells[0] + 1;
    num_nodes_per_dim[1] = num_cells[1] + 1;
    // Set the number of nodes in the z-direction to 1 if dim is 2. The nodes will be 2d
    // if dim==2, but this allows for a unified implementation.
    num_nodes_per_dim[2] = dim == 3 ? num_cells[2] + 1 : 1;

    // The faces will be ordered as follows: First the faces along the x-direction, then
    // the faces along the y-direction. In 3d, the faces along the z-direction will be
    // last.

    // The number of faces along the x and y directions. This is true for 2d, for 3d,
    // the number is adjusted below.
    int num_faces_x = (num_cells[0] + 1) * num_cells[1];
    int num_faces_y = num_cells[0] * (num_cells[1] + 1);
    // In 3d, we also need to count the number of x- and y-faces in a single xy-layer.
    const int num_faces_x_per_xy_layer = num_faces_x;
    const int num_faces_y_per_xy_layer = num_faces_y;

    // Total number of faces in 2d; 3d adjustment is done below.
    int tot_num_faces;
    if (dim == 2)
    {
        tot_num_faces = num_faces_x + num_faces_y;
    }

    if (dim == 3)
    {
        num_faces_x *= num_cells[2];
        num_faces_y *= num_cells[2];
        const int num_faces_z = num_cells[0] * num_cells[1] * (num_cells[2] + 1);
        tot_num_faces = num_faces_x + num_faces_y + num_faces_z;
    }

    // Data structures for node coordinates and face nodes.
    // Define node coordinates as a num_nodes x dim array.
    std::vector<std::vector<double>> nodes(num_nodes, std::vector<double>(dim));
    // We will eventually create a compressed row data storage for the face nodes.
    // However, for convenience store the column indices (the face numbers) in a vector
    // first. Data for the compressed storage will be created later.
    std::vector<int> row_ptr_face_nodes(num_nodes + 1);
    std::vector<int> face_nodes_vector;

    // Let the node indices increase along the x-direction first, then the y-direction.
    // In 3d, the z-direction will be last.

    if (dim == 2)
    {
        for (int j = 0; j < num_nodes_per_dim[1]; ++j)
        {
            for (int i = 0; i < num_nodes_per_dim[0]; ++i)
            {
                const int node_index = i + j * num_nodes_per_dim[0];
                nodes[node_index][0] = x[i];
                nodes[node_index][1] = y[j];

                // Create face nodes
                row_ptr_face_nodes[node_index] = face_nodes_vector.size();
                // Local vector of face indices, to be appended to the global vector.
                std::vector<int> face_nodes_loc;

                if (i > 0)  // There is space for faces to the left.
                {
                    // Add the face to the left in the xy-plane.
                    face_nodes_loc.push_back(i - 1 + j * num_cells[0] + num_faces_x);
                }
                if (i < num_nodes_per_dim[0] - 1)
                {
                    // Add the face to the right in the xy-plane.

                    face_nodes_loc.push_back(i + j * num_cells[0] + num_faces_x);
                }
                if (j > 0)
                {
                    // Add the face below in the xy-plane.
                    face_nodes_loc.push_back(i + (j - 1) * num_nodes_per_dim[0]);
                }
                if (j < num_nodes_per_dim[1] - 1)
                {
                    // Add face above in the xy-plane
                    face_nodes_loc.push_back(i + j * num_nodes_per_dim[0]);
                }
                face_nodes_vector.insert(face_nodes_vector.end(), face_nodes_loc.begin(),
                                         face_nodes_loc.end());
            }
        }
    }
    else  // dim == 3
    {
        for (int k = 0; k < num_nodes_per_dim[2]; ++k)
        {
            for (int j = 0; j < num_nodes_per_dim[1]; ++j)
            {
                for (int i = 0; i < num_nodes_per_dim[0]; ++i)
                {
                    const int node_index = i + j * num_nodes_per_dim[0] +
                                           k * num_nodes_per_dim[0] * num_nodes_per_dim[1];
                    nodes[node_index][0] = x[i];
                    nodes[node_index][1] = y[j];
                    if (dim == 3)
                    {
                        nodes[node_index][2] = z[k];
                    }

                    // Create face nodes
                    row_ptr_face_nodes[node_index] = face_nodes_vector.size();
                    // Local vector of face indices, to be appended to the global vector.
                    std::vector<int> face_nodes_loc;

                    // Faces in the yz-plane.
                    if (j > 0 && k > 0)
                    {
                        face_nodes_loc.push_back(i + (j - 1) * num_nodes_per_dim[0] +
                                                 (k - 1) * num_nodes_per_dim[0] * num_cells[1]);
                    }
                    if (j > 0 && k < num_nodes_per_dim[2] - 1)
                    {
                        face_nodes_loc.push_back(i + (j - 1) * num_nodes_per_dim[0] +
                                                 k * num_nodes_per_dim[0] * num_cells[1]);
                    }
                    if (j < num_nodes_per_dim[1] - 1 && k > 0)
                    {
                        face_nodes_loc.push_back(i + j * num_nodes_per_dim[0] +
                                                 (k - 1) * num_nodes_per_dim[0] * num_cells[1]);
                    }
                    if (j < num_nodes_per_dim[1] - 1 && k < num_nodes_per_dim[2] - 1)
                    {
                        face_nodes_loc.push_back(i + j * num_nodes_per_dim[0] +
                                                 k * num_nodes_per_dim[0] * num_cells[1]);
                    }
                    // Faces in the xz-plane.
                    if (i > 0 && k > 0)
                    {
                        face_nodes_loc.push_back((i - 1) + j * num_cells[0] +
                                                 (k - 1) * num_cells[0] * num_nodes_per_dim[1] +
                                                 num_faces_x);
                    }
                    if (i > 0 && k < num_nodes_per_dim[2] - 1)
                    {
                        face_nodes_loc.push_back((i - 1) + j * num_cells[0] +
                                                 k * num_cells[0] * num_nodes_per_dim[1] +
                                                 num_faces_x);
                    }
                    if (i < num_nodes_per_dim[0] - 1 && k > 0)
                    {
                        face_nodes_loc.push_back(i + j * num_cells[0] +
                                                 (k - 1) * num_cells[0] * num_nodes_per_dim[1] +
                                                 num_faces_x);
                    }
                    if (i < num_nodes_per_dim[0] - 1 && k < num_nodes_per_dim[2] - 1)
                    {
                        face_nodes_loc.push_back(i + j * num_cells[0] +
                                                 k * num_cells[0] * num_nodes_per_dim[1] +
                                                 num_faces_x);
                    }
                    // Faces in the xy-plane.
                    if (i > 0 && j > 0)
                    {
                        face_nodes_loc.push_back((i - 1) + (j - 1) * num_cells[0] +
                                                 k * num_cells[0] * num_cells[1] + num_faces_x +
                                                 num_faces_y);
                    }
                    if (i > 0 && j < num_nodes_per_dim[1] - 1)
                    {
                        face_nodes_loc.push_back((i - 1) + j * num_cells[0] +
                                                 k * num_cells[0] * num_cells[1] + num_faces_x +
                                                 num_faces_y);
                    }
                    if (i < num_nodes_per_dim[0] - 1 && j > 0)
                    {
                        face_nodes_loc.push_back(i + (j - 1) * num_cells[0] +
                                                 k * num_cells[0] * num_cells[1] + num_faces_x +
                                                 num_faces_y);
                    }
                    if (i < num_nodes_per_dim[0] - 1 && j < num_nodes_per_dim[1] - 1)
                    {
                        face_nodes_loc.push_back(i + j * num_cells[0] +
                                                 k * num_cells[0] * num_cells[1] + num_faces_x +
                                                 num_faces_y);
                    }
                    // Append the local face nodes to the global vector.

                    face_nodes_vector.insert(face_nodes_vector.end(), face_nodes_loc.begin(),
                                             face_nodes_loc.end());
                }
            }
        }
    }
    // Set the last element of row_ptr_face_nodes to the size of face_nodes_vector
    row_ptr_face_nodes[num_nodes] = face_nodes_vector.size();

    // Turn the vector into an array
    std::vector<int> col_ptr_face_nodes(face_nodes_vector.begin(), face_nodes_vector.end());

    // The data is an array of ones
    std::vector<int> data_face_nodes(face_nodes_vector.size(), 1);

    auto face_nodes = std::make_shared<CompressedDataStorage<int>>(
        num_nodes, tot_num_faces, row_ptr_face_nodes, col_ptr_face_nodes, data_face_nodes);

    // Create cell faces
    int tot_num_cells = num_cells[0] * num_cells[1];
    tot_num_cells = dim == 3 ? tot_num_cells * num_cells[2] : tot_num_cells;

    std::vector<int> row_ptr(tot_num_faces + 1);
    std::vector<int> col_idx_vector;
    std::vector<int> face_cell_sign_vector;

    int face_index = 0;
    int data_counter = 0;

    const int num_faces_z = (dim == 3) ? num_cells[2] : 1;
    // Create faces along the x and y directions. This loop is common for 2d and 3d, but
    // will do a single iteration in 2d.
    for (int k = 0; k < num_faces_z; ++k)
    {
        // First create the faces along the x-direction. The outer loop is over the
        // y-direction.
        for (int j = 0; j < num_cells[1]; ++j)
        {
            // The first face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point into the first cell.
            face_cell_sign_vector.push_back(-1);
            // The neighboring cell is the one to the right.
            col_idx_vector.push_back(j * num_cells[0] + k * num_cells[0] * num_cells[1]);

            ++face_index;

            // Next loop over the cells in the x-direction. Start at 1 because the first
            // face has been created.
            for (int i = 1; i < num_cells[0]; ++i)
            {
                row_ptr[face_index] = col_idx_vector.size();
                // The normal vector will point out of the first cell.
                face_cell_sign_vector.push_back(1);
                // The neighboring cell is the one to the left.
                col_idx_vector.push_back((i - 1) + j * num_cells[0] +
                                         k * num_cells[0] * num_cells[1]);

                // The normal vector will point into the second cell.
                face_cell_sign_vector.push_back(-1);
                // The neighboring cell is the one to the right.
                col_idx_vector.push_back(i + j * num_cells[0] + k * num_cells[0] * num_cells[1]);

                ++face_index;
            }
            // The last face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point out of the last cell.
            face_cell_sign_vector.push_back(1);
            // The neighboring cell is the one to the left.
            col_idx_vector.push_back((num_cells[0] - 1) + j * num_cells[0] +
                                     k * num_cells[0] * num_cells[1]);
            ++face_index;
        }
    }
    for (int k = 0; k < num_faces_z; ++k)
    {
        // Next create the faces along the y-direction. The outer loop is over the
        // x-direction.
        for (int i = 0; i < num_cells[0]; ++i)
        {
            // The first face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point into the first cell.
            face_cell_sign_vector.push_back(-1);
            // The neighboring cell is the one above.
            col_idx_vector.push_back(i + k * num_cells[0] * num_cells[1]);
            ++face_index;
        }

        // Next loop over the cells in the y-direction. Start at 1 because the first
        // face has been created.
        for (int j = 1; j < num_cells[1]; ++j)
        {
            for (int i = 0; i < num_cells[0]; ++i)
            {
                row_ptr[face_index] = col_idx_vector.size();
                // The normal vector will point out of the first cell.
                face_cell_sign_vector.push_back(1);
                // The neighboring cell is the one below.
                col_idx_vector.push_back(i + (j - 1) * num_cells[0] +
                                         k * num_cells[0] * num_cells[1]);
                // The normal vector will point into the second cell.
                face_cell_sign_vector.push_back(-1);
                // The neighboring cell is the one above.
                col_idx_vector.push_back(i + j * num_cells[0] + k * num_cells[0] * num_cells[1]);
                ++face_index;
            }
        }
        for (int i = 0; i < num_cells[0]; ++i)
        {
            // The first face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point out of this cell.
            face_cell_sign_vector.push_back(1);
            // The neighboring cell is the one above.
            col_idx_vector.push_back(i + (num_cells[1] - 1) * num_cells[0] +
                                     k * num_cells[0] * num_cells[1]);
            ++face_index;
        }
    }
    if (dim == 3)
    {
        // First create faces at the bottom of the domain.
        for (int j = 0; j < num_cells[1]; ++j)
        {
            for (int i = 0; i < num_cells[0]; ++i)
            {
                // First create face at the bottom of the domain.
                row_ptr[face_index] = col_idx_vector.size();
                // The normal vector will point into the first cell.
                face_cell_sign_vector.push_back(-1);
                // The neighboring cell is the one above.
                col_idx_vector.push_back(j * num_cells[0] + i);
                ++face_index;
            }
        }
        // Loop over the cells in the z-direction. Start at 1 because the first
        // face has been created.
        for (int k = 1; k < num_cells[2]; ++k)
        {
            for (int j = 0; j < num_cells[1]; ++j)
            {
                for (int i = 0; i < num_cells[0]; ++i)
                {
                    row_ptr[face_index] = col_idx_vector.size();
                    // The normal vector will point out of the first cell.
                    face_cell_sign_vector.push_back(1);
                    // The neighboring cell is the one below.
                    col_idx_vector.push_back(i + j * num_cells[0] +
                                             (k - 1) * num_cells[0] * num_cells[1]);
                    // The normal vector will point into the second cell.
                    face_cell_sign_vector.push_back(-1);
                    // The neighboring cell is the one above.
                    col_idx_vector.push_back(i + j * num_cells[0] +
                                             k * num_cells[0] * num_cells[1]);
                    ++face_index;
                }
            }
        }
        // Last create faces at the top of the domain.
        for (int j = 0; j < num_cells[1]; ++j)
        {
            for (int i = 0; i < num_cells[0]; ++i)
            {
                // First create face at the bottom of the domain.
                row_ptr[face_index] = col_idx_vector.size();
                // The normal vector will point out of the cell.
                face_cell_sign_vector.push_back(1);
                // The neighboring cell is the one above.
                col_idx_vector.push_back(i + j * num_cells[0] +
                                         num_cells[0] * num_cells[1] * (num_cells[2] - 1));
                ++face_index;
            }
        }
    }
    // Set the last element of row_ptr to the size of col_idx_vector
    row_ptr[tot_num_faces] = col_idx_vector.size();

    auto face_cells = std::make_shared<CompressedDataStorage<int>>(
        tot_num_faces, tot_num_cells, row_ptr, col_idx_vector, face_cell_sign_vector);

    Grid* g = new Grid(dim, std::move(nodes), face_cells, face_nodes);

    return std::unique_ptr<Grid>(g);
}
