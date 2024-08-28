#include <iostream>
#include "Tensor.h"
#include "AutogradOps.h"

int main() {
    std::cout << "Starting main function" << std::endl;

    try {
        auto a = AutogradOps<float>::tensor({2.0}, {1}, true);
        std::cout << "a->data[0] = " << a->data[0] << std::endl;
        auto b = AutogradOps<float>::tensor({3.0}, {1}, true);
        std::cout << "b->data[0] = " << b->data[0] << std::endl;
        auto c = AutogradOps<float>::mul(a, b);
        std::cout << "c->data[0] = " << c->data[0] << std::endl;

        auto d = AutogradOps<float>::tanh(c);
        std::cout << "d->data[0] = " << d->data[0] << std::endl;

        d->backward();

        std::cout << "a->grad->data[0] = " << a->grad->data[0] << std::endl;
        std::cout << "b->grad->data[0] = " << b->grad->data[0] << std::endl;
        std::cout << "c->grad->data[0] = " << c->grad->data[0] << std::endl;
        std::cout << "d->grad->data[0] = " << d->grad->data[0] << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Exiting main function" << std::endl;
    return 0;
}