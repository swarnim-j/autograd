#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "AutogradOps.h"

template<typename T>
void print_tensor(const std::string& name, const std::shared_ptr<Tensor<T>>& tensor) {
    std::cout << name << ": [";
    for (size_t i = 0; i < tensor->data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << tensor->data[i];
    }
    std::cout << "]" << std::endl;
}

void test_binary_case() {
    std::cout << "\n=== Testing Binary Classification Case ===" << std::endl;
    
    // Create predictions (after sigmoid) and targets
    auto predictions = AutogradOps<float>::tensor({0.8f, 0.2f}, {2}, true);  // Confident prediction
    auto targets = AutogradOps<float>::tensor({1.0f, 0.0f}, {2}, false);     // True label is 0

    print_tensor("Predictions", predictions);
    print_tensor("Targets", targets);

    // Compute loss
    auto loss = AutogradOps<float>::cross_entropy_loss(predictions, targets);
    std::cout << "Loss: " << loss->data[0] << std::endl;

    // Compute gradients
    loss->backward();
    print_tensor("Gradients", predictions->grad);

    // Verify gradients manually
    // For binary case, dL/dp = -t/p for positive target, divided by batch size
    float expected_grad_0 = -1.0f / 0.8f / 2.0f;  // -target/prediction for class 0, divided by batch size
    float expected_grad_1 = 0.0f;                  // No gradient for class 1 (target is 0)
    
    assert(std::abs(predictions->grad->data[0] - expected_grad_0) < 1e-4);
    assert(std::abs(predictions->grad->data[1] - expected_grad_1) < 1e-4);
    
    std::cout << "Binary case gradients verified!" << std::endl;
}

void test_multiclass_case() {
    std::cout << "\n=== Testing Multiclass Classification Case ===" << std::endl;
    
    // Create predictions (after softmax) and targets
    auto predictions = AutogradOps<float>::tensor(
        {0.7f, 0.2f, 0.1f,    // First sample (confident prediction)
         0.3f, 0.4f, 0.3f},   // Second sample (uncertain prediction)
        {2, 3}, true);
    
    auto targets = AutogradOps<float>::tensor(
        {1.0f, 0.0f, 0.0f,    // First sample (class 0)
         0.0f, 1.0f, 0.0f},   // Second sample (class 1)
        {2, 3}, false);

    print_tensor("Predictions", predictions);
    print_tensor("Targets", targets);

    // Compute loss
    auto loss = AutogradOps<float>::cross_entropy_loss(predictions, targets);
    std::cout << "Loss: " << loss->data[0] << std::endl;

    // Compute gradients
    loss->backward();
    print_tensor("Gradients", predictions->grad);

    // Verify gradients manually
    // For each sample: dL/dp = -t/p for positive target, divided by batch size
    float expected_grad_00 = -1.0f / 0.7f / 2.0f;  // First sample, class 0 (divided by batch size)
    float expected_grad_11 = -1.0f / 0.4f / 2.0f;  // Second sample, class 1 (divided by batch size)
    
    assert(std::abs(predictions->grad->data[0] - expected_grad_00) < 1e-4);
    assert(std::abs(predictions->grad->data[4] - expected_grad_11) < 1e-4);
    
    std::cout << "Multiclass case gradients verified!" << std::endl;
}

void test_numerical_stability() {
    std::cout << "\n=== Testing Numerical Stability ===" << std::endl;
    
    // Test with very small probabilities
    auto predictions = AutogradOps<float>::tensor({1e-10f, 1.0f - 1e-10f}, {2}, true);
    auto targets = AutogradOps<float>::tensor({1.0f, 0.0f}, {2}, false);

    print_tensor("Predictions (small prob)", predictions);
    
    // This should not produce inf or nan
    auto loss = AutogradOps<float>::cross_entropy_loss(predictions, targets);
    std::cout << "Loss with small probability: " << loss->data[0] << std::endl;
    
    loss->backward();
    print_tensor("Gradients (small prob)", predictions->grad);

    // Check for inf/nan
    assert(!std::isinf(loss->data[0]) && !std::isnan(loss->data[0]));
    assert(!std::isinf(predictions->grad->data[0]) && !std::isnan(predictions->grad->data[0]));
    
    std::cout << "Numerical stability verified!" << std::endl;
}

int main() {
    try {
        test_binary_case();
        test_multiclass_case();
        test_numerical_stability();
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 