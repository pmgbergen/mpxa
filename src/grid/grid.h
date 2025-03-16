#ifndef GRID_GRID_H
#define GRID_GRID_H

#include <algorithm>
#include <vector>

#include "../utils/compressed_storage.h"

class Grid
{
   public:
    Grid(const int dim, double** nodes, CompressedDataStorage<int>* cell_faces,
         CompressedDataStorage<int>* face_nodes);
    ~Grid();

    void compute_geometry();

    // int* boundary_faces();

    const int dim() const;

    // Getters for topological data
    const int num_nodes() const;
    const int num_cells() const;
    const int num_faces() const;
    // int num_boundary_faces();

    // Direct access to the compressed data storage
    const std::vector<int> faces_of_node(const int node) const;
    const std::vector<int> nodes_of_face(const int face) const;
    const std::vector<int> cells_of_face(const int face) const;
    const std::vector<int> faces_of_cell(const int cell) const;
    const int sign_of_face_cell(const int face, const int cell) const;

    // Getters for geometric data
    const double** nodes() const;

    const double** cell_centers() const;
    const double* cell_volumes() const;
    const double* face_areas() const;
    const double** face_normals() const;
    const double** face_centers() const;

    // Also provide access to individual elements
    const double* cell_center(int cell) const;
    const double& cell_volume(int cell) const;
    const double& face_area(int face) const;
    const double* face_normal(int face) const;
    const double* face_center(int face) const;

    // Setters for the geometry data, in case these are computed externally.
    void set_cell_volumes(double* cell_volumes);
    void set_face_areas(double* face_areas);
    void set_face_normals(double** face_normals);
    void set_face_centers(double** face_centers);
    void set_cell_centers(double** cell_centers);

   private:
    int m_dim;

    int m_num_nodes;
    int m_num_cells;
    int m_num_faces;

    CompressedDataStorage<int>* m_cell_faces;
    CompressedDataStorage<int>* m_face_nodes;

    double** m_nodes;
    double* m_cell_volumes;
    double* m_face_areas;
    double** m_face_normals;
    double** m_face_centers;
    double** m_cell_centers;

    int* m_boundary_faces;
};

// Cartesian grid creation
Grid* create_cartesian_grid(const int dim, const int* num_cells, const double* lengths);

#endif  // GRID_GRID_H
