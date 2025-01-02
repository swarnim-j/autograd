#ifndef SGD_H
#define SGD_H

#include <vector>
#include <memory>
#include "Tensor.h"

template<typename T>
class SGD {
public:
    SGD(const std::vector<std::shared_ptr<Tensor<T>>>& parameters, T learning_rate = 0.01)
        : parameters(parameters), learning_rate(learning_rate) {}

    void step() {
        for (auto& param : parameters) {
            if (!param->requires_grad || !param->grad) {
                continue;
            }

            // Basic gradient descent update: param = param - lr * grad
            for (size_t i = 0; i < param->data.size(); ++i) {
                param->data[i] -= learning_rate * param->grad->data[i];
            }
        }
    }

    void zero_grad() {
        for (auto& param : parameters) {
            param->zero_grad();
        }
    }

private:
    std::vector<std::shared_ptr<Tensor<T>>> parameters;
    T learning_rate;
};

#endif // SGD_H 