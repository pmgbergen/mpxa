#ifndef GRID_GRID_H
#define GRID_GRID_H

#include <algorithm>
#include <memory>
#include <vector>
#include <span>

#include "../include/compressed_storage.h"

class Grid
{
   public:
    Grid(const int dim, std::vector<std::vector<double>> nodes,
         std::shared_ptr<CompressedDataStorage<int>> cell_faces,
         std::shared_ptr<CompressedDataStorage<int>> face_nodes);

    ~Grid() = default;  // No need for manual deletion

    static std::unique_ptr<Grid> create_cartesian_grid(const int dim,
                                                       const std::vector<int> num_cells,
                                                       const std::vector<double> lengths);

    void compute_geometry();

    const std::vector<int> boundary_faces() const;

    int dim() const;

    // Getters for topological data
    int num_nodes() const;
    int num_cells() const;
    int num_faces() const;

    // Direct access to the compressed data storage
    const std::span<const int> faces_of_node(const int node) const;
    const std::vector<int> nodes_of_face(const int face) const;
    const std::span<const int> cells_of_face(const int face) const;
    const std::vector<int> faces_of_cell(const int cell) const;
    int sign_of_face_cell(const int face, const int cell) const;
    int num_nodes_of_face(const int face) const;

    const CompressedDataStorage<int>& face_nodes() const;
    const CompressedDataStorage<int>& cell_faces() const;

    // Getters for geometric data
    const std::vector<std::vector<double>>& nodes() const;

    const std::vector<std::vector<double>>& cell_centers() const;
    const std::vector<double>& cell_volumes() const;
    const std::vector<double>& face_areas() const;
    const std::vector<std::vector<double>>& face_normals() const;
    const std::vector<std::vector<double>>& face_centers() const;

    // Also provide access to individual elements
    const std::vector<double>& cell_center(int cell) const;
    const double& cell_volume(int cell) const;
    const double& face_area(int face) const;
    const std::vector<double>& face_normal(int face) const;
    const std::vector<double>& face_center(int face) const;

    // Setters for the geometry data, in case these are computed externally.
    void set_cell_volumes(const std::vector<double>& cell_volumes);
    void set_face_areas(const std::vector<double>& face_areas);
    void set_face_normals(const std::vector<std::vector<double>>& face_normals);
    void set_face_centers(const std::vector<std::vector<double>>& face_centers);
    void set_cell_centers(const std::vector<std::vector<double>>& cell_centers);

   private:
    // compute_geometry() helpers — called in order; each depends on the previous.
    void compute_face_geometry();
    void compute_cell_geometry();
    void fix_normal_orientations();

    int m_dim;

    int m_num_nodes;
    int m_num_cells;
    int m_num_faces;

    std::shared_ptr<CompressedDataStorage<int>> m_cell_faces;
    std::shared_ptr<CompressedDataStorage<int>> m_face_nodes;

    std::vector<std::vector<double>> m_nodes;
    std::vector<double> m_cell_volumes;
    std::vector<double> m_face_areas;
    std::vector<std::vector<double>> m_face_normals;
    std::vector<std::vector<double>> m_face_centers;
    std::vector<std::vector<double>> m_cell_centers;
};

#endif  // GRID_GRID_H
