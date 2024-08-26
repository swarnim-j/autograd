#ifndef ADD_H
#define ADD_H

#include "Tensor.h"

template<typename T>
class Add : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Add<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Add operation requires exactly 2 inputs");
    }
    
    // Save the input tensors for use in backward pass
    this->saved_tensors = inputs;

    // Perform element-wise multiplication
    std::vector<T> result_data(inputs[0]->data.size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        result_data[i] = inputs[0]->data[i] + inputs[1]->data[i];
    }

    // Determine if the result requires gradient computation
    bool requires_grad = inputs[0]->requires_grad || inputs[1]->requires_grad;

    // Determine if the result is a leaf node
    bool is_leaf = !requires_grad;

    // Create the result tensor
    auto result = std::make_shared<Tensor<T>>(
        result_data, 
        inputs[0]->shape, 
        requires_grad,
        is_leaf
    );
        
    // Set the gradient function and save the tensors
    if (result->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }

    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Add<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    auto grad_output = grad_outputs[0];
    auto& a = this->saved_tensors[0];
    auto& b = this->saved_tensors[1];

    std::vector<std::shared_ptr<Tensor<T>>> grads(2, nullptr);
    
    if (a->requires_grad) {
        grads[0] = std::make_shared<Tensor<T>>(std::vector<T>(a->data.size()), a->shape);
        for (size_t i = 0; i < grad_output->data.size(); ++i) {
            grads[0]->data[i] = grad_output->data[i];
        }
    }

    if (b->requires_grad) {
        grads[1] = std::make_shared<Tensor<T>>(std::vector<T>(b->data.size()), b->shape);
        for (size_t i = 0; i < grad_output->data.size(); ++i) {
            grads[1]->data[i] = grad_output->data[i];
        }
    }

    return grads;
}

#endif // ADD_H