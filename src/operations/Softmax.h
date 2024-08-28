#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Tensor.h"
#include "operations/Operation.h"
#include <memory>
#include <cmath>

template<typename T>
class Softmax : public Operation<T> {
public:
    Softmax(int64_t dim = -1) : dim(dim) {}

    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;

    int64_t dim;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Softmax<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Softmax operation requires exactly 1 input");
    }

    this->saved_tensors = inputs;

    // Perform softmax operation
    while (this->dim < 0) {
        this->dim += inputs[0]->shape.size();
    }

    std::vector<T> result_data(inputs[0]->data.size());
    size_t dim_size = inputs[0]->shape[this->dim];
    size_t inner_size = 1;
    size_t outer_size = 1;

    for (size_t i = 0; i < this->dim; ++i) {
        outer_size *= inputs[0]->shape[i];
    }
    for (size_t i = this->dim + 1; i < inputs[0]->shape.size(); ++i) {
        inner_size *= inputs[0]->shape[i];
    }

    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < dim_size; ++j) {
            // Find the maximum value in the dimension
            T max_val = inputs[0]->data[i * inner_size * dim_size + j * inner_size + 0];
            for (size_t k = 1; k < inner_size; ++k) {
                max_val = std::max(max_val, inputs[0]->data[i * inner_size * dim_size + j * inner_size + k]);
            }

            // Compute the sum of the exponentials
            T sum_exp = 0;
            for (size_t k = 0; k < inner_size; ++k) {
                sum_exp += std::exp(inputs[0]->data[i * inner_size * dim_size + j * inner_size + k] - max_val);
            }

            // Normalize the values
            for (size_t k = 0; k < inner_size; ++k) {
                result_data[i * inner_size * dim_size + j * inner_size + k] = std::exp(inputs[0]->data[i * inner_size * dim_size + j * inner_size + k] - max_val) / sum_exp;
            }
        }
    }

    // Determine if the result requires gradient computation
    bool requires_grad = std::any_of(inputs.begin(), inputs.end(), [](const std::shared_ptr<Tensor<T>>& input) {
        return input->requires_grad; 
    });

    // Determine if the result is a leaf node
    bool is_leaf = !requires_grad;

    // Create the result tensor
    auto result = std::make_shared<Tensor<T>>(
        result_data, 
        inputs[0]->shape, 
        requires_grad, 
        is_leaf
    );

    if (result->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }

    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Softmax<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    // TODO: Implement the backward pass
    return {};
}

#endif // SOFTMAX_H