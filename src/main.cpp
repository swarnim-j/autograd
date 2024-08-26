#include <iostream>
#include "Tensor.h"
#include "operations/Operation.h"
#include "AutogradOps.h"

int main() {
    std::cout << "Starting main function" << std::endl;

    try {
        auto a = AutogradOps<float>::tensor({0.21f}, {1}, true);
        auto k = AutogradOps<float>::mul(a, a);
        auto b = AutogradOps<float>::tensor({0.3f}, {1}, true);
        auto c = AutogradOps<float>::add(b, k);
        auto d = AutogradOps<float>::tensor({0.4f}, {1}, true);
        auto e = AutogradOps<float>::mul(c, d);
        auto f = AutogradOps<float>::sigmoid(e);

        std::cout << "e = " << e->data[0] << std::endl;
        std::cout << "d = " << d->data[0] << std::endl;
        std::cout << "c = " << c->data[0] << std::endl;
        std::cout << "b = " << b->data[0] << std::endl;
        std::cout << "a = " << a->data[0] << std::endl;
        std::cout << "k = " << k->data[0] << std::endl;
        std::cout << "f = " << f->data[0] << std::endl;

        f->zero_grad();
        f->backward();

        std::cout << "e.grad = " << e->grad->data[0] << std::endl;
        std::cout << "d.grad = " << d->grad->data[0] << std::endl;
        std::cout << "c.grad = " << c->grad->data[0] << std::endl;
        std::cout << "b.grad = " << b->grad->data[0] << std::endl;
        std::cout << "a.grad = " << a->grad->data[0] << std::endl;
        std::cout << "k.grad = " << k->grad->data[0] << std::endl;
        std::cout << "f.grad = " << f->grad->data[0] << std::endl;
        std::cout << "Exiting main function" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    std::cout << "Exiting main function" << std::endl;
    return 0;
}