#include <iostream>
#include "Tensor.h"
#include "AutogradOps.h"

int main() {
    std::cout << "Starting main function" << std::endl;

    try {
        // Create input tensor
        auto a = AutogradOps<float>::tensor({1.0, 2.0, 4.0, 2}, {4}, true);
        
        // Apply softmax
        auto b = AutogradOps<float>::softmax(a);
        
        // Sum the softmax output to get a scalar
        auto c = AutogradOps<float>::sum(b);
        
        // Print intermediate results
        std::cout << "b->data[0] = " << b->data[0] << std::endl;
        std::cout << "b->data[1] = " << b->data[1] << std::endl;
        std::cout << "b->data[2] = " << b->data[2] << std::endl;
        std::cout << "b->data[3] = " << b->data[3] << std::endl;

        std::cout << "c->data[0] = " << c->data[0] << std::endl;
        
        // Perform backward pass
        c->backward();
        
        // Print gradients
        std::cout << "a->grad->data[0] = " << a->grad->data[0] << std::endl;
        std::cout << "a->grad->data[1] = " << a->grad->data[1] << std::endl;
        std::cout << "a->grad->data[2] = " << a->grad->data[2] << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Exiting main function" << std::endl;
    return 0;
}