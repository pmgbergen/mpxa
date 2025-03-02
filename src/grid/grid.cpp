#include "grid.h"

#include <vector>

Grid::Grid(const int dim, double **nodes, CompressedDataStorage<int> *cell_faces,
           CompressedDataStorage<int> *face_nodes)
    : m_dim(dim), m_nodes(nodes), m_cell_faces(cell_faces), m_face_nodes(face_nodes)
{
    m_num_nodes = m_face_nodes->num_rows();
    m_num_faces = m_face_nodes->num_cols();
    m_num_cells = m_cell_faces->num_cols();
    m_cell_volumes = new double[m_num_cells];
    m_face_areas = new double[m_num_faces];
    m_face_normals = new double *[m_num_faces];
    m_face_centers = new double *[m_num_faces];
    m_cell_centers = new double *[m_num_cells];
    for (int i = 0; i < m_num_faces; i++)
    {
        m_face_normals[i] = new double[m_dim];
        m_face_centers[i] = new double[m_dim];
    }
    for (int i = 0; i < m_num_cells; i++)
    {
        m_cell_centers[i] = new double[m_dim];
    }
}

Grid::~Grid()
{
    delete[] m_cell_volumes;
    delete[] m_face_areas;
    for (int i = 0; i < num_faces(); i++)
    {
        delete[] m_face_normals[i];
        delete[] m_face_centers[i];
    }
    for (int i = 0; i < num_cells(); i++)
    {
        delete[] m_cell_centers[i];
    }

    for (int i = 0; i < m_num_nodes; i++)
    {
        delete[] m_nodes[i];
    }

    delete[] m_face_normals;
    delete[] m_face_centers;
    delete[] m_cell_centers;
    delete[] m_nodes;
    delete m_cell_faces;
    delete m_face_nodes;
}

void Grid::compute_geometry() {}

int *boundary_faces();

// Getters for topological data
int Grid::num_nodes()
{
    return m_num_nodes;
}
int Grid::num_cells()
{
    return m_num_cells;
}
int Grid::num_faces()
{
    return m_num_faces;
}

// Getters for geometric data
const double **Grid::nodes()
{
    return (const double **)m_nodes;
}
const double **Grid::cell_centers()
{
    return (const double **)m_cell_centers;
}
const double *Grid::cell_volumes()
{
    return m_cell_volumes;
}
const double *Grid::face_areas()
{
    return m_face_areas;
}
const double **Grid::face_normals()
{
    return (const double **)m_face_normals;
}
const double **Grid::face_centers()
{
    return (const double **)m_face_centers;
}
// Getters for individual elements
const double *Grid::cell_center(int cell)
{
    return m_cell_centers[cell];
}
const double &Grid::cell_volume(int cell)
{
    return m_cell_volumes[cell];
}
const double &Grid::face_area(int face)
{
    return m_face_areas[face];
}
const double *Grid::face_normal(int face)
{
    return m_face_normals[face];
}
const double *Grid::face_center(int face)
{
    return m_face_centers[face];
}
// Setters for the geometry data, in case these are computed externally.
void Grid::set_cell_volumes(double *cell_volumes)
{
    for (int i = 0; i < m_num_cells; i++)
    {
        m_cell_volumes[i] = cell_volumes[i];
    }
}
void Grid::set_face_areas(double *face_areas)
{
    for (int i = 0; i < m_num_faces; i++)
    {
        m_face_areas[i] = face_areas[i];
    }
}
void Grid::set_face_normals(double **face_normals)
{
    for (int i = 0; i < m_num_faces; i++)
    {
        for (int j = 0; j < m_dim; j++)
        {
            m_face_normals[i][j] = face_normals[i][j];
        }
    }
}
void Grid::set_face_centers(double **face_centers)
{
    for (int i = 0; i < m_num_faces; i++)
    {
        for (int j = 0; j < m_dim; j++)
        {
            m_face_centers[i][j] = face_centers[i][j];
        }
    }
}
void Grid::set_cell_centers(double **cell_centers)
{
    for (int i = 0; i < m_num_cells; i++)
    {
        for (int j = 0; j < m_dim; j++)
        {
            m_cell_centers[i][j] = cell_centers[i][j];
        }
    }
}

// Cartesian grid creation
Grid *create_cartesian_grid(const int dim, const int *num_cells, const double *lengths)
{
    // Dim should be 2 or 3
    if (dim < 2 || dim > 3)
    {
        return nullptr;
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
    int tot_num_faces = num_faces_x + num_faces_y;

    if (dim == 3)
    {
        num_faces_x *= num_cells[2];
        num_faces_y *= num_cells[2];
        const int num_faces_z = num_cells[0] * num_cells[1] * (num_cells[2] + 1);
        tot_num_faces += num_faces_z;
    }

    // Data structures for node coordinates and face nodes.
    // Define node coordinates as a num_nodes x dim array.
    double **nodes = new double *[num_nodes];
    for (int i = 0; i < num_nodes; ++i)
    {
        nodes[i] = new double[dim];
    }
    // We will eventually create a compressed row data storage for the face nodes.
    // However, for convenience store the column indices (the face numbers) in a vector
    // first. Data for the compressed storage will be created later.
    std::vector<int> row_ptr_face_nodes(num_nodes + 1);
    std::vector<int> face_nodes_vector;

    // Let the node indices increase along the x-direction first, then the y-direction.
    // In 3d, the z-direction will be last.
    for (int k = 0; k < num_nodes_per_dim[2]; ++k)
    {
        for (int j = 0; j < num_nodes_per_dim[1]; ++j)
        {
            for (int i = 0; i < num_nodes_per_dim[0]; ++i)
            {
                const int node_index =
                    i + j * num_nodes_per_dim[0] + k * num_nodes_per_dim[0] * num_nodes_per_dim[1];
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

                if (i < num_nodes_per_dim[0] - 1)
                {
                    // Add the face to the right in the xy-plane
                    face_nodes_loc.push_back(node_index + num_faces_x);
                    if (dim == 3 && k > 0)
                    {
                        // Add the face to the right in the xz-plane
                        face_nodes_loc.push_back(node_index + num_faces_x -
                                                 num_faces_y_per_xy_layer);
                    }
                }
                if (i > 0)
                {
                    // Add the face to the left in the xy-plane
                    face_nodes_loc.push_back(node_index - 1);
                    if (dim == 3 && k > 0)
                    {
                        // Add the face to the left in the xz-plane
                        face_nodes_loc.push_back(node_index - 1 - num_faces_y_per_xy_layer);
                    }
                }
                if (j > 0)
                {
                    // Add the face below in the xy-plane
                    face_nodes_loc.push_back(node_index - num_nodes_per_dim[0]);
                    if (dim == 3 && k > 0)
                    {
                        // Add the face below in the xz-plane
                        face_nodes_loc.push_back(node_index - num_nodes_per_dim[0] -
                                                 num_faces_x_per_xy_layer);
                    }
                }
                if (j < num_nodes_per_dim[1] - 1)
                {
                    // Add face above in the xy-plane
                    face_nodes_loc.push_back(node_index);
                    if (dim == 3 && k > 0)
                    {
                        // Add the face above in the xz-plane
                        face_nodes_loc.push_back(node_index - num_faces_x_per_xy_layer);
                    }
                }
                if (dim == 3)
                {
                    if (i < num_cells[0] + 1 && j < num_cells[1] + 1)
                    {
                        // Add the face below in the xz-plane
                        face_nodes_loc.push_back(node_index + num_faces_x_per_xy_layer +
                                                 num_faces_y_per_xy_layer);
                    }
                    if (i < num_cells[0] + 1 && j > 0)
                    {
                        face_nodes_loc.push_back(node_index + num_faces_x_per_xy_layer +
                                                 num_faces_y_per_xy_layer - num_faces_x);
                    }
                    if (i > 0 && j < num_cells[1] + 1)
                    {
                        face_nodes_loc.push_back(node_index + num_faces_x_per_xy_layer +
                                                 num_faces_y_per_xy_layer - 1);
                    }
                    if (i > 0 && j > 0)
                    {
                        face_nodes_loc.push_back(node_index + num_faces_x_per_xy_layer +
                                                 num_faces_y_per_xy_layer - num_faces_x - 1);
                    }
                }
                face_nodes_vector.insert(face_nodes_vector.end(), face_nodes_loc.begin(),
                                         face_nodes_loc.end());
            }
        }
    }
    // Turn the vector into an array
    std::vector<int> col_ptr_face_nodes(face_nodes_vector.begin(), face_nodes_vector.end());

    // The data is an array of ones
    std::vector<int> data_face_nodes(face_nodes_vector.size(), 1);

    CompressedDataStorage<int> *face_nodes = new CompressedDataStorage<int>(
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

                face_cell_sign_vector.push_back(1);
                ++face_index;
            }
            // The last face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point out of the last cell.
            face_cell_sign_vector.push_back(1);
            // The neighboring cell is the one to the left.
            col_idx_vector.push_back((num_cells[1] - 1) + j * num_cells[0] +
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

            // Next loop over the cells in the y-direction. Start at 1 because the first
            // face has been created.
            for (int j = 1; j < num_cells[1]; ++j)
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
            // The last face has a single neighboring cell.
            row_ptr[face_index] = col_idx_vector.size();
            // The normal vector will point out of the last cell.
            face_cell_sign_vector.push_back(1);
            // The neighboring cell is the one below.
            col_idx_vector.push_back(i + (num_cells[1] * num_cells[0]) +
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
                                         num_cells[0] * num_cells[1] * num_cells[2]);
                ++face_index;
            }
        }
    }
    std::vector<int> col_idx(col_idx_vector.begin(), col_idx_vector.end());
    std::vector<int> face_cell_sign(face_cell_sign_vector.begin(), face_cell_sign_vector.end());
    CompressedDataStorage<int> *face_cells = new CompressedDataStorage<int>(
        tot_num_faces, tot_num_cells, row_ptr, col_idx, face_cell_sign);

    Grid *grid = new Grid(dim, nodes, face_cells, face_nodes);

    // Clean up temporary arrays
    // No need to delete x, y, z, num_nodes_per_dim, row_ptr_face_nodes, col_ptr_face_nodes,
    // data_face_nodes, row_ptr, col_idx, face_cell_sign as they are now vectors

    // TODO: Add geometry computation
    return grid;
}
