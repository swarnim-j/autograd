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
private:
    std::shared_ptr<Tensor<T>> input;  // Store input tensor
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Softmax<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Softmax operation requires exactly 1 input");
    }

    this->input = inputs[0];  // Store input tensor

    // Ensure dim is non-negative
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
        for (size_t k = 0; k < inner_size; ++k) {
            // Find the maximum value in the dimension
            T max_val = inputs[0]->data[i * dim_size * inner_size + k];
            for (size_t j = 1; j < dim_size; ++j) {
                max_val = std::max(max_val, inputs[0]->data[i * dim_size * inner_size + j * inner_size + k]);
            }

            // Compute the sum of the exponentials
            T sum_exp = 0;
            for (size_t j = 0; j < dim_size; ++j) {
                sum_exp += std::exp(inputs[0]->data[i * dim_size * inner_size + j * inner_size + k] - max_val);
            }

            // Normalize the values
            for (size_t j = 0; j < dim_size; ++j) {
                size_t idx = i * dim_size * inner_size + j * inner_size + k;
                result_data[idx] = std::exp(inputs[0]->data[idx] - max_val) / sum_exp;
            }
        }
    }

    // Create the result tensor
    auto result = std::make_shared<Tensor<T>>(
        result_data, 
        inputs[0]->shape, 
        inputs[0]->requires_grad,
        false  // Not a leaf node since it's an operation output
    );

    // Set the gradient function if input requires gradient
    if (inputs[0]->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }

    // Save the output tensor for backward pass
    this->saved_tensors = {result};

    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Softmax<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    if (grad_outputs.size() != 1 || this->saved_tensors.size() != 1) {
        throw std::invalid_argument("Softmax backward pass requires exactly 1 gradient output and 1 saved output tensor");
    }

    auto& grad_output = grad_outputs[0];
    auto& output = this->saved_tensors[0];  // This is the softmax output

    // For scalar input, the gradient is 0 since softmax(x) = 1 for any x
    if (this->input->is_scalar()) {
        return {std::make_shared<Tensor<T>>(
            std::vector<T>{0},
            std::vector<size_t>{1},
            this->input->requires_grad,
            false  // Not a leaf node
        )};
    }

    std::vector<T> grad_input_data(this->input->data.size());
    size_t dim_size = this->input->shape[this->dim];
    size_t inner_size = 1;
    size_t outer_size = 1;

    for (size_t i = 0; i < this->dim; ++i) {
        outer_size *= this->input->shape[i];
    }
    for (size_t i = this->dim + 1; i < this->input->shape.size(); ++i) {
        inner_size *= this->input->shape[i];
    }

    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t k = 0; k < inner_size; ++k) {
            T sum_grad_times_output = 0;
            for (size_t j = 0; j < dim_size; ++j) {
                size_t idx = i * dim_size * inner_size + j * inner_size + k;
                sum_grad_times_output += grad_output->data[idx] * output->data[idx];
            }

            for (size_t j = 0; j < dim_size; ++j) {
                size_t idx = i * dim_size * inner_size + j * inner_size + k;
                grad_input_data[idx] = output->data[idx] * (grad_output->data[idx] - sum_grad_times_output);
            }
        }
    }

    return {std::make_shared<Tensor<T>>(
        grad_input_data,
        this->input->shape,
        this->input->requires_grad,
        false  // Not a leaf node
    )};
}

#endif // SOFTMAX_H