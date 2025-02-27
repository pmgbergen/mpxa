#include "grid.h"

Grid::Grid(int dim, double **nodes, CompressedDataStorage<int> *cell_faces,
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
