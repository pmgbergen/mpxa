#ifndef GRID_GRID_H
#define GRID_GRID_H

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <vector>

#include "../include/compressed_storage.h"

class Grid
{
   public:
    Grid(int dim, std::vector<double> nodes,
         std::shared_ptr<CompressedDataStorage<int>> cell_faces,
         std::shared_ptr<CompressedDataStorage<int>> face_nodes);
    Grid(int dim, const double* nodes, std::size_t nodes_size,
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
    std::span<const double> nodes() const;
    std::span<const double> cell_centers() const;
    std::span<const double> cell_volumes() const;
    std::span<const double> face_areas() const;
    std::span<const double> face_normals() const;
    std::span<const double> face_centers() const;

    // Also provide access to individual elements
    std::array<double, 3> node(int node) const;
    std::array<double, 3> cell_center(int cell) const;
    double cell_volume(int cell) const;
    double face_area(int face) const;
    std::array<double, 3> face_normal(int face) const;
    std::array<double, 3> face_center(int face) const;

    // Setters for the geometry data, in case these are computed externally.
    void set_cell_volumes(const double* data, std::size_t size);
    void set_face_areas(const double* data, std::size_t size);
    void set_face_normals(const double* data, std::size_t size);
    void set_face_centers(const double* data, std::size_t size);
    void set_cell_centers(const double* data, std::size_t size);

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

    std::vector<double> m_nodes_owned;
    std::span<const double> m_nodes_view;
    std::vector<double> m_cell_volumes_owned;
    std::span<const double> m_cell_volumes_view;
    std::vector<double> m_face_areas_owned;
    std::span<const double> m_face_areas_view;
    std::vector<double> m_face_normals_owned;
    std::span<const double> m_face_normals_view;
    std::vector<double> m_face_centers_owned;
    std::span<const double> m_face_centers_view;
    std::vector<double> m_cell_centers_owned;
    std::span<const double> m_cell_centers_view;
};

#endif  // GRID_GRID_H
