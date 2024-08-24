#include "Tensor.hpp"
#include <iostream>
#include <memory>

int main() {
    std::cout << "Autograd Engine Example" << std::endl;

    // Create two 2x2 Tensors
    std::shared_ptr<Tensor<int>> t1 = std::make_shared<Tensor<int>>(std::vector<int>{1, 2, 3, 4}, std::vector<size_t>{2, 2});
    std::shared_ptr<Tensor<int>> t2 = std::make_shared<Tensor<int>>(std::vector<int>{5, 6, 7, 8}, std::vector<size_t>{2, 2});

    // Print the original tensors
    std::cout << "Tensor 1:" << std::endl;
    t1->print();

    std::cout << "Tensor 2:" << std::endl;
    t2->print();

    // Perform element-wise addition
    std::shared_ptr<Tensor<int>> t3 = *t1 + t2;
    std::cout << "Tensor 1 + Tensor 2:" << std::endl;
    t3->print();

    // Perform element-wise subtraction
    std::shared_ptr<Tensor<int>> t4 = *t1 - t2;
    std::cout << "Tensor 1 - Tensor 2:" << std::endl;
    t4->print();

    // Perform element-wise multiplication
    std::shared_ptr<Tensor<int>> t5 = *t1 * t2;
    std::cout << "Tensor 1 * Tensor 2:" << std::endl;
    t5->print();

    return 0;
}
