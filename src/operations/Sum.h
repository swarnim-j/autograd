#ifndef SUM_H
#define SUM_H

#include "Tensor.h"
#include "operations/Operation.h"
#include <memory>
#include <numeric>

template<typename T>
class Sum : public Operation<T> {
public:
    Sum(int64_t dim = -1) : dim(dim) {}

    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;

    int64_t dim;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Sum<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Sum operation requires exactly 1 input");
    }

    this->saved_tensors = inputs;

    // Ensure dim is non-negative
    while (this->dim < 0) {
        this->dim += inputs[0]->shape.size();
    }

    std::vector<size_t> result_shape = inputs[0]->shape;
    result_shape.erase(result_shape.begin() + this->dim);

    size_t dim_size = inputs[0]->shape[this->dim];
    size_t inner_size = 1;
    size_t outer_size = 1;

    for (size_t i = 0; i < this->dim; ++i) {
        outer_size *= inputs[0]->shape[i];
    }
    for (size_t i = this->dim + 1; i < inputs[0]->shape.size(); ++i) {
        inner_size *= inputs[0]->shape[i];
    }

    std::vector<T> result_data(outer_size * inner_size, 0);

    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t k = 0; k < inner_size; ++k) {
            T sum = 0;
            for (size_t j = 0; j < dim_size; ++j) {
                size_t idx = i * dim_size * inner_size + j * inner_size + k;
                sum += inputs[0]->data[idx];
            }
            result_data[i * inner_size + k] = sum;
        }
    }

    // Determine if the result requires gradient computation
    bool requires_grad = inputs[0]->requires_grad;

    // Determine if the result is a leaf node
    bool is_leaf = !requires_grad;

    // Create the result tensor
    auto result = std::make_shared<Tensor<T>>(
        result_data, 
        result_shape, 
        requires_grad,
        is_leaf
    );
    
    // Set the gradient function
    if (result->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }

    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Sum<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    if (grad_outputs.size() != 1 || this->saved_tensors.size() != 1) {
        throw std::invalid_argument("Sum backward pass requires exactly 1 gradient output and 1 saved input tensor");
    }

    auto& grad_output = grad_outputs[0];
    auto& input = this->saved_tensors[0];

    std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;
    grad_inputs.reserve(input->shape[this->dim]);

    // Calculate the size of dimensions before and after the summed dimension
    size_t pre_dim_size = 1;
    size_t post_dim_size = 1;
    for (size_t i = 0; i < this->dim; ++i) {
        pre_dim_size *= input->shape[i];
    }
    for (size_t i = this->dim + 1; i < input->shape.size(); ++i) {
        post_dim_size *= input->shape[i];
    }

    // Create a tensor for each element in the summed dimension
    for (size_t j = 0; j < input->shape[this->dim]; ++j) {
        std::vector<T> grad_input_data(input->data.size() / input->shape[this->dim], 0);

        for (size_t i = 0; i < pre_dim_size; ++i) {
            for (size_t k = 0; k < post_dim_size; ++k) {
                size_t grad_output_idx = i * post_dim_size + k;
                size_t input_idx = (i * input->shape[this->dim] + j) * post_dim_size + k;
                grad_input_data[i * post_dim_size + k] = grad_output->data[grad_output_idx];
            }
        }

        std::vector<size_t> grad_input_shape = input->shape;
        grad_input_shape.erase(grad_input_shape.begin() + this->dim);

        auto grad_input = std::make_shared<Tensor<T>>(
            grad_input_data,
            grad_input_shape,
            input->requires_grad,
            false  // Not a leaf node
        );

        grad_inputs.push_back(grad_input);
    }

    return grad_inputs;
}

#endif // SUM_H