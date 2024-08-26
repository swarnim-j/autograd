#ifndef SUM_H
#define SUM_H

#include "Tensor.h"
#include "Operation.h"
#include <memory>

template<typename T>
class Sum : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Sum<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("Sum operation requires at least 1 input");
    }

    // Save the input tensors for use in backward pass
    this->saved_tensors = inputs;

    // Perform summation
    std::vector<T> result_data(inputs[0]->data.size(), 0);
    for (const auto& input : inputs) {
        for (size_t i = 0; i < result_data.size(); ++i) {
            result_data[i] += input->data[i];
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
    
    // Set the gradient function and save the tensors
    if (result->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }

    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Sum<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    auto grad_output = grad_outputs[0];

    std::vector<std::shared_ptr<Tensor<T>>> grads(this->saved_tensors.size(), nullptr);
    
    for (size_t i = 0; i < this->saved_tensors.size(); ++i) {
        if (this->saved_tensors[i]->requires_grad) {
            grads[i] = std::make_shared<Tensor<T>>(std::vector<T>(this->saved_tensors[i]->data.size()), this->saved_tensors[i]->shape);
            for (size_t j = 0; j < grad_output->data.size(); ++j) {
                grads[i]->data[j] = grad_output->data[j]; // Each input contributes equally to the output
            }
        }
    }

    return grads;
}

#endif // SUM_H