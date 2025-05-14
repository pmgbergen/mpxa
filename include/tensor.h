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

    const std::vector<double>& isotropic_data() const;
    const std::vector<const double*> diagonal_data() const;
    const std::vector<const double*> full_data() const;

   private:
    const int m_dim;
    const int m_num_cells;

    bool m_is_isotropic;
    bool m_is_diagonal;

    std::vector<double> m_k_xx;
    std::vector<double> m_k_yy;
    std::vector<double> m_k_xy;
    std::vector<double> m_k_zz;
    std::vector<double> m_k_xz;
    std::vector<double> m_k_yz;
    std::vector<double> m_zeros;

    mutable std::vector<const double*> m_diagonal_data;
    mutable std::vector<const double*> m_full_data;
};

#endif  // TENSOR_H