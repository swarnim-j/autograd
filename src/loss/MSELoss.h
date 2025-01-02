#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "Tensor.h"
#include "operations/Operation.h"
#include <memory>

template<typename T>
class MSELoss : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("MSELoss requires exactly 2 inputs (predictions and targets)");
        }

        auto& predictions = inputs[0];
        auto& targets = inputs[1];

        if (predictions->data.size() != targets->data.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        this->saved_tensors = {predictions, targets};

        T loss = 0;
        size_t n = predictions->data.size();
        
        for (size_t i = 0; i < n; ++i) {
            T diff = predictions->data[i] - targets->data[i];
            loss += diff * diff;
        }
        loss /= n;

        return {std::make_shared<Tensor<T>>(
            std::vector<T>{loss},
            std::vector<size_t>{1},
            predictions->requires_grad,
            false  // Not a leaf node
        )};
    }

    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("MSELoss backward pass requires exactly 1 gradient output");
        }

        auto& predictions = this->saved_tensors[0];
        auto& targets = this->saved_tensors[1];
        auto& grad_output = grad_outputs[0];

        size_t n = predictions->data.size();
        std::vector<T> grad_input(n);

        // Compute gradient: 2(pred - target)/n * grad_output
        T grad_scale = 2.0 * grad_output->data[0] / n;
        for (size_t i = 0; i < n; ++i) {
            grad_input[i] = (predictions->data[i] - targets->data[i]) * grad_scale;
        }

        // Return gradients for both inputs (prediction gradient and target gradient)
        return {
            std::make_shared<Tensor<T>>(grad_input, predictions->shape, predictions->requires_grad, false),
            std::make_shared<Tensor<T>>(std::vector<T>(n, 0), targets->shape, false, false)  // Target gradient is always zero
        };
    }
};

#endif // MSE_LOSS_H 