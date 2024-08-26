#ifndef AUTOGRAD_OPS_H
#define AUTOGRAD_OPS_H

#include "operations/Mul.h"
#include "operations/Add.h"

template<typename T>
class AutogradOps {
public:
    static std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& input1, const std::shared_ptr<Tensor<T>>& input2);
    static std::shared_ptr<Tensor<T>> tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad = false);
};

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
std::shared_ptr<Tensor<T>> AutogradOps<T>::tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad) {
    return std::make_shared<Tensor<T>>(data, shape, requires_grad, true);
}

#endif // AUTOGRAD_OPS_H