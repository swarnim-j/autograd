#ifndef AUTOGRAD_OPS_H
#define AUTOGRAD_OPS_H

#include "operations/Mul.h"
#include "operations/Add.h"
#include "operations/ReLU.h"
#include "operations/Sigmoid.h"
#include "operations/Softmax.h"
#include "operations/Tanh.h"
#include "operations/Linear.h"
#include "loss/MSELoss.h"
#include "loss/CrossEntropyLoss.h"

template<typename T>
class AutogradOps {
public:
    static std::shared_ptr<Tensor<T>> tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad = false);
    static std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& input);
    static std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& input);
    static std::shared_ptr<Tensor<T>> softmax(const std::shared_ptr<Tensor<T>>& input, int64_t dim = -1);
    static std::shared_ptr<Tensor<T>> tanh(const std::shared_ptr<Tensor<T>>& input);
    static std::shared_ptr<Tensor<T>> linear(const std::shared_ptr<Tensor<T>>& input, size_t in_features, size_t out_features, bool use_bias = true);
    static std::shared_ptr<Tensor<T>> mse_loss(const std::shared_ptr<Tensor<T>>& predictions, const std::shared_ptr<Tensor<T>>& targets);
    static std::shared_ptr<Tensor<T>> cross_entropy_loss(const std::shared_ptr<Tensor<T>>& predictions, const std::shared_ptr<Tensor<T>>& targets);
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
std::shared_ptr<Tensor<T>> AutogradOps<T>::softmax(const std::shared_ptr<Tensor<T>>& input, int64_t dim) {
    auto op = std::make_shared<Softmax<T>>(dim);
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::tanh(const std::shared_ptr<Tensor<T>>& input) {
    auto op = std::make_shared<Tanh<T>>();
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::linear(const std::shared_ptr<Tensor<T>>& input, size_t in_features, size_t out_features, bool use_bias) {
    auto op = std::make_shared<Linear<T>>(in_features, out_features, use_bias);
    return op->forward({input})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::mse_loss(const std::shared_ptr<Tensor<T>>& predictions, const std::shared_ptr<Tensor<T>>& targets) {
    auto op = std::make_shared<MSELoss<T>>();
    return op->forward({predictions, targets})[0];
}

template<typename T>
std::shared_ptr<Tensor<T>> AutogradOps<T>::cross_entropy_loss(const std::shared_ptr<Tensor<T>>& predictions, const std::shared_ptr<Tensor<T>>& targets) {
    auto op = std::make_shared<CrossEntropyLoss<T>>();
    return op->forward({predictions, targets})[0];
}

#endif // AUTOGRAD_OPS_H