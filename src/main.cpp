#include <iostream>
#include "Tensor.h"
#include "AutogradOps.h"

int main() {
    std::cout << "Starting main function" << std::endl;

    try {
        auto a = AutogradOps<float>::tensor({2}, {1}, true);
        auto b = AutogradOps<float>::tensor({3}, {1}, true);
        auto c = AutogradOps<float>::add(a, b);
        
        c->backward();

        std::cout << "a->data[0] = " << a->data[0] << std::endl;
        std::cout << "b->data[0] = " << b->data[0] << std::endl;
        std::cout << "c->data[0] = " << c->data[0] << std::endl;

        std::cout << "a->grad->data[0] = " << a->grad->data[0] << std::endl;
        std::cout << "b->grad->data[0] = " << b->grad->data[0] << std::endl;
        std::cout << "c->grad->data[0] = " << c->grad->data[0] << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Exiting main function" << std::endl;
    return 0;
}