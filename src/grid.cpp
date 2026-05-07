#include "../include/grid.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
constexpr int SPATIAL_DIM = 3;

void validate_geometry_size(std::size_t size, std::size_t expected, const char* name)
{
    if (size != expected)
    {
        throw std::invalid_argument(std::string(name) + " size mismatch.");
    }
}

inline double coord(std::span<const double> values, int num_entities, int coord_ind, int entity_ind)
{
    return values[coord_ind * num_entities + entity_ind];
}

inline std::array<double, SPATIAL_DIM> entity_as_array(std::span<const double> values,
                                                       int num_entities, int entity_ind)
{
    std::array<double, SPATIAL_DIM> out{0.0, 0.0, 0.0};
    for (int coord_ind = 0; coord_ind < SPATIAL_DIM; ++coord_ind)
    {
        out[coord_ind] = coord(values, num_entities, coord_ind, entity_ind);
    }
    return out;
}
}  // namespace

Grid::Grid(int dim, std::vector<double> nodes,
           std::shared_ptr<CompressedDataStorage<int>> cell_faces,
           std::shared_ptr<CompressedDataStorage<int>> face_nodes)
    : m_dim(dim),
      m_cell_faces(std::move(cell_faces)),
      m_face_nodes(std::move(face_nodes)),
      m_nodes_owned(std::move(nodes)),
      m_nodes_view(m_nodes_owned)
{
    m_num_nodes = m_face_nodes->num_rows();
    m_num_faces = m_face_nodes->num_cols();
    m_num_cells = m_cell_faces->num_cols();

    validate_geometry_size(m_nodes_view.size(), SPATIAL_DIM * m_num_nodes, "nodes");

    m_cell_volumes_owned.assign(m_num_cells, 0.0);
    m_cell_volumes_view = m_cell_volumes_owned;
    m_face_areas_owned.assign(m_num_faces, 0.0);
    m_face_areas_view = m_face_areas_owned;
    m_face_normals_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_normals_view = m_face_normals_owned;
    m_face_centers_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_centers_view = m_face_centers_owned;
    m_cell_centers_owned.assign(SPATIAL_DIM * m_num_cells, 0.0);
    m_cell_centers_view = m_cell_centers_owned;
}

Grid::Grid(int dim, const double* nodes, std::size_t nodes_size,
           std::shared_ptr<CompressedDataStorage<int>> cell_faces,
           std::shared_ptr<CompressedDataStorage<int>> face_nodes)
    : m_dim(dim),
      m_cell_faces(std::move(cell_faces)),
      m_face_nodes(std::move(face_nodes)),
      m_nodes_view(nodes, nodes_size)
{
    m_num_nodes = m_face_nodes->num_rows();
    m_num_faces = m_face_nodes->num_cols();
    m_num_cells = m_cell_faces->num_cols();

    validate_geometry_size(m_nodes_view.size(), SPATIAL_DIM * m_num_nodes, "nodes");

    m_cell_volumes_owned.assign(m_num_cells, 0.0);
    m_cell_volumes_view = m_cell_volumes_owned;
    m_face_areas_owned.assign(m_num_faces, 0.0);
    m_face_areas_view = m_face_areas_owned;
    m_face_normals_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_normals_view = m_face_normals_owned;
    m_face_centers_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_centers_view = m_face_centers_owned;
    m_cell_centers_owned.assign(SPATIAL_DIM * m_num_cells, 0.0);
    m_cell_centers_view = m_cell_centers_owned;
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

int Grid::dim() const
{
    return m_dim;
}

// Getters for topological data
int Grid::num_nodes() const
{
    return m_num_nodes;
}
int Grid::num_cells() const
{
    return m_num_cells;
}
int Grid::num_faces() const
{
    return m_num_faces;
}

const CompressedDataStorage<int>& Grid::face_nodes() const
{
    return *m_face_nodes;
}

const CompressedDataStorage<int>& Grid::cell_faces() const
{
    return *m_cell_faces;
}

const std::span<const int> Grid::faces_of_node(const int node) const
{
    return m_face_nodes->cols_in_row(node);
}

const std::vector<int> Grid::nodes_of_face(const int face) const
{
    return m_face_nodes->rows_in_col(face);
}

const std::span<const int> Grid::cells_of_face(const int face) const
{
    return m_cell_faces->cols_in_row(face);
}

const std::vector<int> Grid::faces_of_cell(const int cell) const
{
    return m_cell_faces->rows_in_col(cell);
}

int Grid::sign_of_face_cell(const int face, const int cell) const
{
    return m_cell_faces->value(face, cell);
}

int Grid::num_nodes_of_face(const int face) const
{
    return m_face_nodes->rows_in_col(face).size();
}

// Getters for geometric data
std::span<const double> Grid::nodes() const
{
    return m_nodes_view;
}
std::span<const double> Grid::cell_centers() const
{
    return m_cell_centers_view;
}
std::span<const double> Grid::cell_volumes() const
{
    return m_cell_volumes_view;
}
std::span<const double> Grid::face_areas() const
{
    return m_face_areas_view;
}
std::span<const double> Grid::face_normals() const
{
    return m_face_normals_view;
}
std::span<const double> Grid::face_centers() const
{
    return m_face_centers_view;
}
// Getters for individual elements
std::array<double, 3> Grid::node(int node) const
{
    return entity_as_array(m_nodes_view, m_num_nodes, node);
}
std::array<double, 3> Grid::cell_center(int cell) const
{
    return entity_as_array(m_cell_centers_view, m_num_cells, cell);
}
double Grid::cell_volume(int cell) const
{
    return m_cell_volumes_view[cell];
}
double Grid::face_area(int face) const
{
    return m_face_areas_view[face];
}
std::array<double, 3> Grid::face_normal(int face) const
{
    return entity_as_array(m_face_normals_view, m_num_faces, face);
}
std::array<double, 3> Grid::face_center(int face) const
{
    return entity_as_array(m_face_centers_view, m_num_faces, face);
}
// Setters for the geometry data, in case these are computed externally.
void Grid::set_cell_volumes(const double* data, std::size_t size)
{
    validate_geometry_size(size, m_num_cells, "cell_volumes");
    m_cell_volumes_view = std::span<const double>(data, size);
}
void Grid::set_face_areas(const double* data, std::size_t size)
{
    validate_geometry_size(size, m_num_faces, "face_areas");
    m_face_areas_view = std::span<const double>(data, size);
}
void Grid::set_face_normals(const double* data, std::size_t size)
{
    validate_geometry_size(size, SPATIAL_DIM * m_num_faces, "face_normals");
    m_face_normals_view = std::span<const double>(data, size);
}
void Grid::set_face_centers(const double* data, std::size_t size)
{
    validate_geometry_size(size, SPATIAL_DIM * m_num_faces, "face_centers");
    m_face_centers_view = std::span<const double>(data, size);
}
void Grid::set_cell_centers(const double* data, std::size_t size)
{
    validate_geometry_size(size, SPATIAL_DIM * m_num_cells, "cell_centers");
    m_cell_centers_view = std::span<const double>(data, size);
}

void Grid::compute_geometry()
{
    m_face_normals_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_normals_view = m_face_normals_owned;
    m_face_centers_owned.assign(SPATIAL_DIM * m_num_faces, 0.0);
    m_face_centers_view = m_face_centers_owned;
    m_cell_centers_owned.assign(SPATIAL_DIM * m_num_cells, 0.0);
    m_cell_centers_view = m_cell_centers_owned;
    m_face_areas_owned.assign(m_num_faces, 0.0);
    m_face_areas_view = m_face_areas_owned;
    m_cell_volumes_owned.assign(m_num_cells, 0.0);
    m_cell_volumes_view = m_cell_volumes_owned;

    compute_face_geometry();
    compute_cell_geometry();
    fix_normal_orientations();
}

void Grid::compute_face_geometry()
{
    // Loop over all faces, get the nodes of the face, compute the face center and area
    // from the node coordinates. The loop does not work very well with the storage format
    // for node-face relation (which is a mapping from nodes to faces), but it will have
    // to do for now.
    for (int i{0}; i < num_faces(); ++i)
    {
        std::vector<int> loc_nodes = nodes_of_face(i);
        const int num_nodes = loc_nodes.size();

        for (int j{0}; j < SPATIAL_DIM; ++j)
        {
            double sum = 0.0;
            for (int k{0}; k < num_nodes; ++k)
            {
                sum += coord(m_nodes_view, m_num_nodes, j, loc_nodes[k]);
            }
            m_face_centers_owned[j * m_num_faces + i] = sum / num_nodes;
        }

        if (m_dim == 2)
        {
            const double dx = coord(m_nodes_view, m_num_nodes, 0, loc_nodes[1]) -
                              coord(m_nodes_view, m_num_nodes, 0, loc_nodes[0]);
            const double dy = coord(m_nodes_view, m_num_nodes, 1, loc_nodes[1]) -
                              coord(m_nodes_view, m_num_nodes, 1, loc_nodes[0]);
            m_face_areas_owned[i] = std::sqrt(dx * dx + dy * dy);
            m_face_normals_owned[0 * m_num_faces + i] = dy;
            m_face_normals_owned[1 * m_num_faces + i] = -dx;
        }
        else  // m_dim == 3
        {
            if (num_nodes == 3)
            {
                std::array<double, SPATIAL_DIM> v1{0.0, 0.0, 0.0};
                std::array<double, SPATIAL_DIM> v2{0.0, 0.0, 0.0};
                for (int j{0}; j < SPATIAL_DIM; ++j)
                {
                    v1[j] = coord(m_nodes_view, m_num_nodes, j, loc_nodes[1]) -
                            coord(m_nodes_view, m_num_nodes, j, loc_nodes[0]);
                    v2[j] = coord(m_nodes_view, m_num_nodes, j, loc_nodes[2]) -
                            coord(m_nodes_view, m_num_nodes, j, loc_nodes[0]);
                }
                std::array<double, SPATIAL_DIM> normal{0.0, 0.0, 0.0};
                normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
                normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
                normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
                m_face_areas_owned[i] =
                    0.5 * std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                    normal[2] * normal[2]);
                for (int j{0}; j < SPATIAL_DIM; ++j)
                {
                    m_face_normals_owned[j * m_num_faces + i] = normal[j] / m_face_areas_owned[i];
                }
            }
            else  // num_nodes == 4
            {
                // NOTE: Valid only for axis-aligned quadrilateral faces.
                // Use bounding-box extents to compute area and normal direction.
                auto min_coords = node(loc_nodes[0]);
                auto max_coords = min_coords;
                for (int j{1}; j < num_nodes; ++j)
                {
                    const auto loc_node = node(loc_nodes[j]);
                    for (int k{0}; k < SPATIAL_DIM; ++k)
                    {
                        min_coords[k] = std::min(min_coords[k], loc_node[k]);
                        max_coords[k] = std::max(max_coords[k], loc_node[k]);
                    }
                }
                for (int j{0}; j < SPATIAL_DIM; ++j)
                {
                    if (min_coords[j] == max_coords[j])
                    {
                        const int k = (j + 1) % SPATIAL_DIM;
                        const int l = (j + 2) % SPATIAL_DIM;
                        m_face_areas_owned[i] =
                            (max_coords[k] - min_coords[k]) * (max_coords[l] - min_coords[l]);
                        m_face_normals_owned[j * m_num_faces + i] = m_face_areas_owned[i];
                        break;
                    }
                }
            }
        }
    }
}

void Grid::compute_cell_geometry()
{
    // Loop over all cells, compute cell center (average of face centers) and cell volume
    // (divergence theorem: sum of face-area × distance-to-cell-center / dim).
    // Requires face centers and face areas to already be computed.
    for (int i{0}; i < m_num_cells; ++i)
    {
        std::vector<int> loc_faces = faces_of_cell(i);
        const int num_faces = loc_faces.size();

        for (int j{0}; j < SPATIAL_DIM; ++j)
        {
            double sum = 0.0;
            for (int k{0}; k < num_faces; ++k)
            {
                sum += coord(m_face_centers_view, m_num_faces, j, loc_faces[k]);
            }
            m_cell_centers_owned[j * m_num_cells + i] = sum / num_faces;
        }

        m_cell_volumes_owned[i] = 0.0;
        for (int j{0}; j < num_faces; ++j)
        {
            double dist = 0.0;
            for (int k{0}; k < SPATIAL_DIM; ++k)
            {
                const double face_to_cell = coord(m_cell_centers_view, m_num_cells, k, i) -
                                            coord(m_face_centers_view, m_num_faces, k,
                                                  loc_faces[j]);
                dist += face_to_cell * coord(m_face_normals_view, m_num_faces, k, loc_faces[j]) /
                        m_face_areas_view[loc_faces[j]];
            }
            m_cell_volumes_owned[i] += m_face_areas_view[loc_faces[j]] * std::abs(dist) / m_dim;
        }
    }
}

void Grid::fix_normal_orientations()
{
    // Ensure that each face normal points away from the cell for which sign_of_face_cell > 0.
    // Requires cell centers to already be computed.
    for (int i{0}; i < m_num_faces; ++i)
    {
        const auto loc_cells = cells_of_face(i);

        double dot_prod = 0.0;
        for (int j{0}; j < SPATIAL_DIM; ++j)
        {
            const double face_to_cell = coord(m_face_centers_view, m_num_faces, j, i) -
                                        coord(m_cell_centers_view, m_num_cells, j, loc_cells[0]);
            dot_prod += face_to_cell * coord(m_face_normals_view, m_num_faces, j, i);
        }

        if ((sign_of_face_cell(i, loc_cells[0]) < 0 && dot_prod > 0) ||
            (sign_of_face_cell(i, loc_cells[0]) > 0 && dot_prod < 0))
        {
            for (int j{0}; j < SPATIAL_DIM; ++j)
            {
                m_face_normals_owned[j * m_num_faces + i] *= -1;
            }
        }
    }
}

// Cartesian grid creation

static std::unique_ptr<Grid> construct_grid_1d(const int num_cells, const double length)
{
    // NOTE: This function is not tested, so should be used with caution.
    const int num_nodes = num_cells + 1;
    const int dim = 1;
    const double dx = length / num_cells;
    std::vector<double> nodes(SPATIAL_DIM * num_nodes, 0.0);

    // Filling in nodes.
    for (int i = 0; i < num_nodes; ++i)
    {
        nodes[0 * num_nodes + i] = dx * i;
    }

    // Constructing face - cells mapping. shape=(num_faces, num_cells)
    std::vector<int> face_cells_row_ptr(num_nodes + 1);
    std::vector<int> face_cells_col_idx;
    std::vector<int> face_cells_sign_vector;
    face_cells_col_idx.reserve((num_nodes - 1) * 2);
    face_cells_sign_vector.reserve((num_nodes - 1) * 2);
    for (int i = 0; i < num_nodes; ++i)
    {
        face_cells_row_ptr[i + 1] = face_cells_row_ptr[i];
        // Add left cell
        if (i != 0)
        {
            face_cells_col_idx.push_back(i - 1);
            face_cells_row_ptr[i + 1] += 1;
            face_cells_sign_vector.push_back(1);
        }
        // Add right cell
        if (i != (num_nodes - 1))
        {
            face_cells_col_idx.push_back(i);
            face_cells_row_ptr[i + 1] += 1;
            face_cells_sign_vector.push_back(-1);
        }
    }

    auto face_cells = std::make_shared<CompressedDataStorage<int>>(
        num_nodes, num_cells, std::move(face_cells_row_ptr), std::move(face_cells_col_idx),
        std::move(face_cells_sign_vector));

    // Constructing face - nodes mapping (identity). shape=(num_nodes, num_faces)
    std::vector<int> face_nodes_row_ptr(num_nodes + 1);
    std::vector<int> face_nodes_col_idx(num_nodes);
    std::vector<int> face_nodes_values(num_nodes, 1);
    std::iota(face_nodes_row_ptr.begin(), face_nodes_row_ptr.end(), 0);
    std::iota(face_nodes_col_idx.begin(), face_nodes_col_idx.end(), 0);
    auto face_nodes = std::make_shared<CompressedDataStorage<int>>(
        num_nodes, num_nodes, std::move(face_nodes_row_ptr), std::move(face_nodes_col_idx),
        std::move(face_nodes_values));

    return std::make_unique<Grid>(1, std::move(nodes), face_cells, face_nodes);
}

std::unique_ptr<Grid> Grid::create_cartesian_grid(const int dim, const std::vector<int> num_cells,
                                                  const std::vector<double> lengths)
{
    if (dim == 1)
    {
        return construct_grid_1d(num_cells[0], lengths[0]);
    }

    // Dim should be 2 or 3
    if (dim < 2 || dim > 3)
    {
        throw std::invalid_argument("Invalid dimension: dim must be 2 or 3.");
    }
    if (dim != num_cells.size())
    {
        throw std::invalid_argument("num_cells dimensions must correspond to 'dim'.");
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
    std::vector<double> nodes(SPATIAL_DIM * num_nodes, 0.0);
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
                nodes[0 * num_nodes + node_index] = x[i];
                nodes[1 * num_nodes + node_index] = y[j];

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
                    nodes[0 * num_nodes + node_index] = x[i];
                    nodes[1 * num_nodes + node_index] = y[j];
                    if (dim == 3)
                    {
                        nodes[2 * num_nodes + node_index] = z[k];
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
        num_nodes, tot_num_faces, std::move(row_ptr_face_nodes), std::move(col_ptr_face_nodes), std::move(data_face_nodes), true);

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
        tot_num_faces, tot_num_cells, std::move(row_ptr), std::move(col_idx_vector), std::move(face_cell_sign_vector), true);

    Grid* g = new Grid(dim, std::move(nodes), face_cells, face_nodes);

    return std::unique_ptr<Grid>(g);
}
