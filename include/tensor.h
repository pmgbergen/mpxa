#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

class SecondOrderTensor
{
   public:
    SecondOrderTensor(const int dim, const int num_cells, const std::vector<double>& k_xx);
    ~SecondOrderTensor();

    // Setters for optional tensor components in 2d
    SecondOrderTensor& with_kyy(const std::vector<double>& k_yy);
    SecondOrderTensor& with_kxy(const std::vector<double>& k_xy);
    // Setters for optional tensor components in 3d
    SecondOrderTensor& with_kzz(const std::vector<double>& k_zz);
    SecondOrderTensor& with_kxz(const std::vector<double>& k_xz);
    SecondOrderTensor& with_kyz(const std::vector<double>& k_yz);

    // Get properties of the tensor
    bool is_isotropic() const;
    bool is_diagonal() const;

    const int dim() const;

    // Return the isotropic value for a given cell
    double isotropic_data(int cell) const;
    // Return the diagonal values for a given cell (size = dim)
    std::vector<double> diagonal_data(
        int cell) const;  // returns size 3: [xx, yy, zz], zero-padded for 2D
    // Return the full tensor values for a given cell (size = dim*dim)
    std::vector<double> full_data(
        int cell) const;  // returns size 6: [xx, yy, zz, xy, xz, yz], zero-padded for 2D

   private:
    int m_dim;
    int m_num_cells;

    bool m_is_isotropic;
    bool m_is_diagonal;

    std::vector<double> m_k_xx;
    std::vector<double> m_k_yy;
    std::vector<double> m_k_xy;
    std::vector<double> m_k_zz;
    std::vector<double> m_k_xz;
    std::vector<double> m_k_yz;
    std::vector<double> m_zeros;
};

#endif  // TENSOR_H