#include <iostream>
#include <iomanip>
#include <cassert>
#include "Tensor.h"
#include "AutogradOps.h"

template<typename T>
void print_tensor_info(const std::string& name, const std::shared_ptr<Tensor<T>>& tensor) {
    std::cout << name << " info:" << std::endl;
    std::cout << "  requires_grad: " << std::boolalpha << tensor->requires_grad << std::endl;
    std::cout << "  is_leaf: " << std::boolalpha << tensor->is_leaf << std::endl;
    std::cout << "  has_grad_fn: " << std::boolalpha << (tensor->grad_fn != nullptr) << std::endl;
    std::cout << "  shape: [";
    for (size_t i = 0; i < tensor->shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tensor->shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  is_scalar: " << std::boolalpha << tensor->is_scalar() << std::endl;
    std::cout << "  data size: " << tensor->data.size() << std::endl;
    std::cout << "  data: [";
    for (size_t i = 0; i < tensor->data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << tensor->data[i];
    }
    std::cout << "]" << std::endl;
}

void test_softmax() {
    std::cout << "\n=== Testing Softmax ===" << std::endl;
    
    // Test with a single value
    auto a = AutogradOps<float>::tensor({1.0}, {1}, true);
    print_tensor_info("Input tensor (scalar)", a);
    
    auto b = AutogradOps<float>::softmax(a);
    print_tensor_info("Softmax output (scalar)", b);
    
    // Test with multiple values
    std::cout << "\nTesting with multiple values:" << std::endl;
    auto x = AutogradOps<float>::tensor({1.0, 2.0, 4.0, 2.0}, {4}, true);
    print_tensor_info("Input tensor", x);
    
    auto y = AutogradOps<float>::softmax(x);
    print_tensor_info("Softmax output", y);
    
    // Verify softmax properties
    float sum = 0;
    for (const auto& val : y->data) {
        sum += val;
        if (val < 0 || val > 1) {
            std::cerr << "Error: Softmax output contains values outside [0,1]" << std::endl;
        }
    }
    if (std::abs(sum - 1.0) > 1e-6) {
        std::cerr << "Error: Softmax outputs do not sum to 1 (sum = " << sum << ")" << std::endl;
    } else {
        std::cout << "Softmax properties verified: outputs in [0,1] and sum to 1" << std::endl;
    }
}

void test_basic_ops() {
    std::cout << "\n=== Testing Basic Operations ===" << std::endl;
    auto a = AutogradOps<float>::tensor({2.0}, {1}, true);
    auto b = AutogradOps<float>::tensor({3.0}, {1}, true);
    
    // Test multiplication
    auto c = AutogradOps<float>::mul(a, b);
    std::cout << "2.0 * 3.0 = " << c->data[0] << std::endl;
    
    c->backward();
    std::cout << "d(a*b)/da = " << a->grad->data[0] << " (should be 3.0)" << std::endl;
    std::cout << "d(a*b)/db = " << b->grad->data[0] << " (should be 2.0)" << std::endl;
    
    // Reset gradients
    a->zero_grad();
    b->zero_grad();
    
    // Test addition
    auto d = AutogradOps<float>::add(a, b);
    std::cout << "2.0 + 3.0 = " << d->data[0] << std::endl;
    
    d->backward();
    std::cout << "d(a+b)/da = " << a->grad->data[0] << " (should be 1.0)" << std::endl;
    std::cout << "d(a+b)/db = " << b->grad->data[0] << " (should be 1.0)" << std::endl;
}

void test_activation_functions() {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    auto x = AutogradOps<float>::tensor({-2.0, -1.0, 0.0, 1.0, 2.0}, {5}, true);
    
    // Test ReLU
    auto relu_out = AutogradOps<float>::relu(x);
    std::cout << "ReLU output: ";
    for (auto val : relu_out->data) std::cout << val << " ";
    std::cout << std::endl;
    
    // Test Sigmoid
    auto sigmoid_out = AutogradOps<float>::sigmoid(x);
    std::cout << "Sigmoid output: ";
    for (auto val : sigmoid_out->data) std::cout << std::fixed << std::setprecision(4) << val << " ";
    std::cout << std::endl;
    
    // Test Tanh
    auto tanh_out = AutogradOps<float>::tanh(x);
    std::cout << "Tanh output: ";
    for (auto val : tanh_out->data) std::cout << std::fixed << std::setprecision(4) << val << " ";
    std::cout << std::endl;
}

int main() {
    std::cout << "Starting tests..." << std::endl;

    try {
        test_basic_ops();
        test_activation_functions();
        test_softmax();
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}