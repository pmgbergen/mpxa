#include "grid.h"


Grid::Grid(int dim, double** nodes, CompressedDataStorage<int>* cell_faces, CompressedDataStorage<int>* face_nodes)
    : m_dim(dim)
    , m_nodes(nodes)
    , m_cell_faces(cell_faces)
    , m_face_nodes(face_nodes)
    {
        m_num_nodes = m_face_nodes->num_rows();
        m_num_faces = m_face_nodes->num_cols();
        m_num_cells = m_cell_faces->num_cols();
        m_cell_volumes = new double[m_num_cells];
        m_face_areas = new double[m_num_faces];
        m_face_normals = new double*[m_num_faces];
        m_face_centers = new double*[m_num_faces];
        m_cell_centers = new double*[m_num_cells];
        for (int i = 0; i < m_num_faces; i++) {
            m_face_normals[i] = new double[m_dim];
            m_face_centers[i] = new double[m_dim];
        }
        for (int i = 0; i < m_num_cells; i++) {
            m_cell_centers[i] = new double[m_dim];
        }

    }
    
Grid::~Grid() {

    delete[] m_cell_volumes;
    delete[] m_face_areas;
    for (int i = 0; i < num_faces(); i++) {
        delete[] m_face_normals[i];
        delete[] m_face_centers[i];
    }


    delete[] m_face_normals;
    delete[] m_face_centers;
    delete[] m_cell_volumes;
    delete[] m_face_areas;
    delete[] m_cell_centers;
}

void Grid::compute_geometry() {

}

int* boundary_faces();

// Getters for topological data
int Grid::num_nodes() {
    return m_num_nodes;
}
int Grid::num_cells() {
    return m_num_cells;
}
int Grid::num_faces() {
    return m_num_faces;
}
int num_boundary_faces();

// Direct access to the compressed data storage
const int* cell_faces(int cell);
const int* face_nodes(int face);
// Data that can be accessed through the compressed data storage.