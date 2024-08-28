#ifndef AUTOGRAD_OPS_H
#define AUTOGRAD_OPS_H

#include "operations/Mul.h"
#include "operations/Add.h"
#include "operations/ReLU.h"
#include "operations/Sigmoid.h"
#include "operations/Sum.h"
#include "operations/Softmax.h"
#include "operations/Tanh.h"

template<typename T>
class AutogradOps {
public:
    static std::shared_ptr<Tensor<T>> tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad);
    static std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& input);
    static std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& input);
    static std::shared_ptr<Tensor<T>> sum(const std::vector<std::shared_ptr<Tensor<T>>>& inputs);
    static std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& input, int dim = -1);
    static std::shared_ptr<Tensor<T>> tanh(const std::shared_ptr<Tensor<T>>& input);
};

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor<T>>(data, shape, requires_grad, true);
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::mul(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2) {
    auto op = std::make_shared<Mul<T>>();
    return op->forward({input1, input2})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::add(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2) {
    auto op = std::make_shared<Add<T>>();
    return op->forward({input1, input2})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::relu(const std::shared_ptr<Tensor<T>>& input) {
    auto op = std::make_shared<ReLU<T>>();
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::sigmoid(const std::shared_ptr<Tensor<T>>& input) {
    auto op = std::make_shared<Sigmoid<T>>();
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::sum(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    auto op = std::make_shared<Sum<T>>();
    return op->forward(inputs)[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::softmax(const std::shared_ptr<Tensor<T>>& input, int dim) {
    auto op = std::make_shared<Softmax<T>>(dim);
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::tanh(const std::shared_ptr<Tensor<T>>& input) {
    auto op = std::make_shared<Tanh<T>>();
    return op->forward({input})[0];
}

#endif // AUTOGRAD_OPS_H