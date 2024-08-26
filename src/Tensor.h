#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <functional>

#include "Operation.h"

template<typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    std::vector<T> data; // Tensor data
    std::vector<size_t> shape; // Tensor shape
    std::shared_ptr<Tensor<T>> grad; // Gradient tensor
    std::shared_ptr<Operation<T>> grad_fn; // Gradient function
    bool is_leaf; // Whether the tensor is a leaf node
    bool requires_grad; // Whether the tensor requires gradient computation

    void backward(); // Backward pass
    void zero_grad(); // Set all gradients to zero

    // Constructors
    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad = false, bool is_leaf = true);

    // Check if tensor is scalar
    bool is_scalar() const;
};

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
}

template<typename T>
void Tensor<T>::backward() {
    if (!this->is_scalar()) {
        throw std::runtime_error("backward() can only be called on scalar tensors");
    }
    if (!this->grad_fn) {
        throw std::runtime_error("Cannot call backward on a tensor without grad_fn");
    }
    if (!this->grad) {
        this->grad = std::make_shared<Tensor<T>>(std::vector<T>(data.size(), 1), shape);
    }

    std::vector<std::shared_ptr<Tensor<T>>> grad_outputs = this->grad_fn->backward({this->grad});

    for (size_t i = 0; i < this->grad_fn->saved_tensors.size(); ++i) {
        auto& saved_tensor = this->grad_fn->saved_tensors[i];
        if (saved_tensor->requires_grad) {
            if (!saved_tensor->grad) {
                saved_tensor->grad = grad_outputs[i];
            } else {
                // Accumulate gradients
                for (size_t j = 0; j < saved_tensor->grad->data.size(); ++j) {
                    saved_tensor->grad->data[j] += grad_outputs[i]->data[j];
                }
            }
            
            // Only call backward if there's a grad_fn
            if (saved_tensor->grad_fn) {
                saved_tensor->backward();
            }
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
bool Tensor<T>::is_scalar() const {
    return data.size() == 1;
}

#endif // TENSOR_H