#ifndef RELU_H
#define RELU_H

#include "Tensor.h"
#include "Operation.h"
#include <memory>

template<typename T>
class ReLU : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> ReLU<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLU operation requires exactly 1 input");
    }

    // Save the input tensor for use in backward pass
    this->saved_tensors = inputs;

    // Perform ReLU operation
    std::vector<T> result_data(inputs[0]->data.size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        result_data[i] = std::max(static_cast<T>(0), inputs[0]->data[i]);
    }

    // Create the result tensor
    auto result = std::make_shared<Tensor<T>>(
        result_data, 
        inputs[0]->shape, 
        inputs[0]->requires_grad,
        !inputs[0]->requires_grad
    );

    // Set the gradient function and save the tensors
    if (result->requires_grad) {
        result->grad_fn = this->shared_from_this();
    }
    
    return {result};
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> ReLU<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    auto grad_output = grad_outputs[0];
    auto& input = this->saved_tensors[0];

    std::vector<std::shared_ptr<Tensor<T>>> grads(1, nullptr);

    if (input->requires_grad) {
        grads[0] = std::make_shared<Tensor<T>>(std::vector<T>(input->data.size()), input->shape);
        for (size_t i = 0; i < grad_output->data.size(); ++i) {
            grads[0]->data[i] = (input->data[i] > 0) ? grad_output->data[i] : 0;
        }
    }

    return grads;
}

#endif // RELU_H
