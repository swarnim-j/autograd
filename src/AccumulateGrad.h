#ifndef ACCUMULATE_GRAD_H
#define ACCUMULATE_GRAD_H

#include "Tensor.h"

template<typename T>
class AccumulateGrad : public Operation<T> {
public:
    std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> AccumulateGrad<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("AccumulateGrad operation requires exactly 1 input");
    }
    this->save_tensors(inputs);
    this->set_next_functions(inputs);
    return inputs;
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> AccumulateGrad<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    auto& var = this->saved_tensors[0];

    if (var->grad) {
        for (size_t i = 0; i < var->grad->data.size(); ++i) {
            var->grad->data[i] += grad_outputs[0]->data[i];
        }
    } else {
        var->grad = std::make_shared<Tensor<T>>(grad_outputs[0]->data, grad_outputs[0]->shape);
    }
    return {};
}

#endif // ACCUMULATE_GRAD_H
