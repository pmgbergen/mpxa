#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <span>
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
    inline double isotropic_data(int cell) const noexcept
    {
        return m_k_full[cell * DATA_PER_CELL];
    }
    // Return the diagonal values for a given cell (size = dim)
    inline std::span<const double, 3> diagonal_data(int cell) const noexcept
    {
        return std::span<const double, 3>(&m_k_full[cell * DATA_PER_CELL], 3);
    }

    // Return the full tensor values for a given cell (size = dim*dim)
    inline std::span<const double, 6> full_data(int cell) const noexcept
    {
        return std::span<const double, 6>(&m_k_full[cell * DATA_PER_CELL], DATA_PER_CELL);
    }

   private:
    int m_dim;
    int m_num_cells;

    bool m_is_isotropic;
    bool m_is_diagonal;

    std::vector<double> m_k_full;

    static constexpr size_t K_XX_OFFSET = 0;
    static constexpr size_t K_YY_OFFSET = 1;
    static constexpr size_t K_ZZ_OFFSET = 2;
    static constexpr size_t K_XY_OFFSET = 3;
    static constexpr size_t K_XZ_OFFSET = 4;
    static constexpr size_t K_YZ_OFFSET = 5;
    static constexpr size_t DATA_PER_CELL = 6;
};

#endif  // TENSOR_H