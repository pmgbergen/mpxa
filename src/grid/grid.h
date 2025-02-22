#ifndef GRID_GRID_H
#define GRID_GRID_H

#include "../utils/compressed_storage.h"

class Grid {
    public:
        Grid(int dim, double** nodes, CompressedDataStorage<int>* cell_faces, CompressedDataStorage<int>* face_nodes);
        ~Grid();

        void compute_geometry();

        int* boundary_faces();

        // Getters for topological data
        int num_nodes();
        int num_cells();
        int num_faces();
        int num_boundary_faces();

        // Direct access to the compressed data storage
        const int* cell_faces(int cell);
        const int* face_nodes(int face);
        // Data that can be accessed through the compressed data storage.
        // TODO: It may be desirable to accept arrays of indices as arguments to these functions.
        const int* face_cells(int face);
        const int* cell_nodes(int cell);
        const int* node_faces(int node);
        const int* node_cells(int node);

        // Getters for geometric data
        const double** nodes();

        const double** cell_centers();
        const double* cell_volumes();
        const double* face_areas();
        const double** face_normals();
        const double** face_centers();
        
        // Also provide access to individual elements
        const double* cell_center(int cell);
        const double& cell_volume(int cell);
        const double& face_area(int face);
        const double* face_normal(int face);
        const double* face_center(int face);

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
#endif  // GRID_GRID_H
