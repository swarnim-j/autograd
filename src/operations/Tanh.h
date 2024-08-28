#ifndef TANH_H
#define TANH_H

#include "Tensor.h"
#include "operations/Operation.h"

template<typename T>
class Tanh : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override;
};

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> Tanh<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Tanh operation requires exactly 1 input");
    }

    // Save the input tensor for use in backward pass
    this->saved_tensors = inputs;

    // Perform tanh operation
    auto result_data = std::vector<T>(inputs[0]->data.size());
    std::transform(inputs[0]->data.begin(), inputs[0]->data.end(), result_data.begin(), [](T value) {
        return std::tanh(value);
    });

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
std::vector<std::shared_ptr<Tensor<T>>> Tanh<T>::backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) {
    auto grad_output = grad_outputs[0];
    auto& input = this->saved_tensors[0];

    std::vector<std::shared_ptr<Tensor<T>>> grads(1, nullptr);

    if (input->requires_grad) {
        grads[0] = std::make_shared<Tensor<T>>(std::vector<T>(input->data.size()), input->shape);
        for (size_t i = 0; i < grad_output->data.size(); ++i) {
            T tanh_value = std::tanh(input->data[i]);
            grads[0]->data[i] = grad_output->data[i] * (1 - tanh_value * tanh_value);
        }
    }

    return grads;
}
    

#endif // TANH_H