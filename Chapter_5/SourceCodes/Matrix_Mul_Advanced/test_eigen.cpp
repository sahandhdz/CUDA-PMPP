#include <iostream>
#include <Eigen/Dense>

// g++ -I/libs/eigen test_eigen.cpp -o test

int main(){
    // Eigen::Vector3d a(1.0, 2.0, 3.0);
    // Eigen::Vector3d b(4.0, 5.0, 6.0);
    // double dot_product = a.dot(b);

    // std::cout << "Dot product: " << dot_product << std::endl;

    // Create two random matrices: A (3x2) and B (2x4)
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 2);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(2, 4);

    // Multiply A and B -> result will be 3x4
    Eigen::MatrixXd C = A * B;

    // Print all matrices
    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "A * B =\n" << C << "\n";

    return 0;
}