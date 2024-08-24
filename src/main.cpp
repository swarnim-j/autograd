#include "Tensor.hpp"
#include "Graph.hpp"
#include <iostream>
#include <memory>
#include <functional>

// Helper function to print a tensor
template <typename T>
void print_tensor(const std::string& name, const std::shared_ptr<Tensor<T>>& tensor) {
    std::cout << name << ":" << std::endl;
    tensor->print();
    std::cout << std::endl;
}

int main() {
    std::cout << "Autograd Engine Test" << std::endl;

    // Create a graph
    Graph<float> graph;

    // Create input tensors
    auto x = graph.add_input(std::make_shared<Tensor<float>>(std::vector<float>{1.0f, 2.0f, 3.0f}, std::vector<size_t>{3}));
    auto y = graph.add_input(std::make_shared<Tensor<float>>(std::vector<float>{4.0f, 5.0f, 6.0f}, std::vector<size_t>{3}));

    print_tensor("x", x->forward());
    print_tensor("y", y->forward());

    // Define operations
    auto add_op = graph.add_operation({x, y}, [](const std::vector<std::shared_ptr<Tensor<float>>>& inputs) {
        return *inputs[0] + inputs[1];
    });

    auto mult_op = graph.add_operation({x, y}, [](const std::vector<std::shared_ptr<Tensor<float>>>& inputs) {
        return *inputs[0] * inputs[1];
    });

    // Perform computations
    auto z_add = graph.forward(add_op);
    print_tensor("z_add (x + y)", z_add);

    auto z_mult = graph.forward(mult_op);
    print_tensor("z_mult (x * y)", z_mult);

    // More complex operation: (x + y) * x
    auto complex_op = graph.add_operation({add_op, x}, [](const std::vector<std::shared_ptr<Tensor<float>>>& inputs) {
        return *inputs[0] * inputs[1];
    });

    auto result = graph.forward(complex_op);
    print_tensor("result ((x + y) * x)", result);

    return 0;
}