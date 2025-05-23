#include "../include/multipoint_common.h"

BasisConstructor::BasisConstructor(const int dim)
    : m_dim(dim), m_coord_matrix(), m_basis_matrix(), m_rhs_matrix()
{
    // Initialize the matrices with appropriate sizes
    m_coord_matrix = MatrixXd::Zero(dim, dim);
    // The first column of the coord matrix will be ones.
    for (int i = 0; i <= dim; ++i)
    {
        m_coord_matrix(i, 0) = 1.0;
    }

    m_basis_matrix = MatrixXd::Zero(dim, dim);
    m_rhs_matrix = MatrixXd::Identity(dim, dim);
}

std::vector<std::vector<double>> BasisConstructor::compute_basis_functions(
    const std::vector<std::array<double, 3>>& coords)
{
    // Compute the basis functions and their gradients. The implementation is highly
    // uncertain and will depend on the specific problem.
    // This is a placeholder for the actual implementation.

    for (int i = 0; i <= m_dim; ++i)
    {
        for (int j = 1; j <= m_dim; ++j)
        {
            m_basis_matrix(i, j) = coords[i][j];
        }
    }

    // Solve the linear system with m_coord_matrix as the left-hand side and
    // m_rhs_matrix as the right-hand side.
    Eigen::PartialPivLU<MatrixXd> lu(m_coord_matrix);
    if (lu.isInvertible())
    {
        m_basis_matrix = lu.solve(m_rhs_matrix);
    }
    else
    {
        throw std::runtime_error("Matrix is not invertible.");
    }
    // Store the computed basis functions in the output vector.
    std::vector<std::vector<double>> basis_functions(m_dim + 1, std::vector<double>(m_dim));
    for (int i = 0; i <= m_dim; ++i)
    {
        for (int j = 0; j < m_dim; ++j)
        {
            // Store the computed basis functions in the output vector. The first column
            // of the basis functions is the constant term, which we ignore.
            basis_functions[i][j] = m_basis_matrix(i, j + 1);
        }
    }
    return basis_functions;
}