#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <functional>

template<typename T>
class Operation;

template<typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    std::vector<T> data; // Tensor data
    std::vector<size_t> shape; // Tensor shape
    std::vector<size_t> strides; // Tensor strides
    std::shared_ptr<Tensor<T>> grad; // Gradient tensor
    std::shared_ptr<Operation<T>> grad_fn; // Gradient function
    bool is_leaf; // Whether the tensor is a leaf node
    bool requires_grad; // Whether the tensor requires gradient computation

    void backward(); // Backward pass
    void zero_grad(); // Set all gradients to zero

    // Constructors
    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad = false, bool is_leaf = true);

    // Check if tensor is scalar
    bool is_scalar() const {
        // A tensor is scalar if it has shape {1} or empty shape
        return shape.empty() || (shape.size() == 1 && shape[0] == 1);
    }

private:
    void compute_strides(); // Compute tensor strides
};

#include "operations/Operation.h"

// constructor
template<typename T>
Tensor<T>::Tensor(
    const std::vector<T>& data, 
    const std::vector<size_t>& shape,
    bool requires_grad,
    bool is_leaf
) : data(data), shape(shape), requires_grad(requires_grad), is_leaf(is_leaf), grad(nullptr), grad_fn(nullptr) {

    // Check if the shape matches the data size
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    if (data.size() != total_size) {
        throw std::invalid_argument("Data size does not match the shape dimensions.");
    }

    compute_strides();
}

template<typename T>
void Tensor<T>::backward() {
    if (!this->requires_grad) {
        throw std::runtime_error("Calling backward on a tensor that doesn't require gradients");
    }
    if (!this->is_scalar()) {
        throw std::runtime_error("backward() can only be called on scalar tensors");
    }
    if (!this->grad_fn && !this->is_leaf) {
        throw std::runtime_error("Cannot call backward on a non-leaf tensor that doesn't have a grad_fn");
    }

    // Initialize gradient if not already done
    if (!this->grad) {
        // For scalar tensors, initialize gradient as 1.0
        this->grad = std::make_shared<Tensor<T>>(
            std::vector<T>{1},
            std::vector<size_t>{1},
            false,  // requires_grad
            true   // is_leaf
        );
    }

    // If there's no grad_fn (leaf node), we're done
    if (!this->grad_fn) {
        return;
    }

    std::vector<std::shared_ptr<Tensor<T>>> grad_outputs = this->grad_fn->backward({this->grad});

    // Validate gradient outputs
    if (grad_outputs.size() != this->grad_fn->saved_tensors.size()) {
        throw std::runtime_error("Number of gradient outputs doesn't match number of inputs");
    }

    for (size_t i = 0; i < this->grad_fn->saved_tensors.size(); ++i) {
        auto& saved_tensor = this->grad_fn->saved_tensors[i];
        if (!saved_tensor->requires_grad) {
            continue;
        }

        auto& grad_output = grad_outputs[i];
        if (grad_output->data.size() != saved_tensor->data.size()) {
            throw std::runtime_error("Gradient data size mismatch");
        }

        if (!saved_tensor->grad) {
            saved_tensor->grad = grad_output;
        } else {
            // Accumulate gradients
            if (saved_tensor->grad->data.size() != grad_output->data.size()) {
                throw std::runtime_error("Accumulated gradient data size mismatch");
            }
            for (size_t j = 0; j < saved_tensor->grad->data.size(); ++j) {
                saved_tensor->grad->data[j] += grad_output->data[j];
            }
        }
        
        // Only call backward if there's a grad_fn
        if (saved_tensor->grad_fn) {
            saved_tensor->backward();
        }
    }
}

template<typename T>
void Tensor<T>::zero_grad() {
    if (this->requires_grad && this->grad) {
        this->grad = nullptr;
    }

    if (this->grad_fn) {
        for (auto& saved_tensor : this->grad_fn->saved_tensors) {
            saved_tensor->zero_grad();
        }
    }
}

template<typename T>
void Tensor<T>::compute_strides() {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    this->strides.resize(shape.size());
    size_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        this->strides[i] = stride;
        stride *= this->shape[i];
    }
}

#endif // TENSOR_H