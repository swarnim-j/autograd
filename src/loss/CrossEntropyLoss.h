#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "Tensor.h"
#include "operations/Operation.h"
#include <memory>
#include <cmath>
#include <numeric>

template<typename T>
class CrossEntropyLoss : public Operation<T> {
public:
    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("CrossEntropyLoss requires exactly 2 inputs (predictions and targets)");
        }

        auto& predictions = inputs[0];
        auto& targets = inputs[1];

        if (predictions->data.size() != targets->data.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }

        // Save inputs for backward pass
        this->saved_tensors = {predictions, targets};

        size_t batch_size = predictions->shape[0];
        size_t num_classes = predictions->data.size() / batch_size;

        T loss = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_classes; ++j) {
                size_t idx = i * num_classes + j;
                if (targets->data[idx] > 0) {  // Only for non-zero target probabilities
                    // Add small epsilon to prevent log(0)
                    T pred = std::max(predictions->data[idx], static_cast<T>(1e-7));
                    pred = std::min(pred, static_cast<T>(1.0 - 1e-7));
                    loss -= std::log(pred) * targets->data[idx];
                }
            }
        }
        
        loss /= batch_size;  // Average over batch

        auto result = std::make_shared<Tensor<T>>(
            std::vector<T>{loss},
            std::vector<size_t>{1},
            predictions->requires_grad,
            false  // Not a leaf node
        );
        result->grad_fn = this->shared_from_this();
        return {result};
    }

    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("CrossEntropyLoss backward pass requires exactly 1 gradient output");
        }

        auto& predictions = this->saved_tensors[0];
        auto& targets = this->saved_tensors[1];
        auto& grad_output = grad_outputs[0];

        size_t batch_size = predictions->shape[0];
        size_t num_classes = predictions->data.size() / batch_size;

        // Gradient is -target/(prediction) * grad_output/batch_size
        std::vector<T> grad_input(predictions->data.size());
        T scale = grad_output->data[0] / batch_size;

        for (size_t i = 0; i < grad_input.size(); ++i) {
            T pred = std::max(predictions->data[i], static_cast<T>(1e-7));
            pred = std::min(pred, static_cast<T>(1.0 - 1e-7));
            grad_input[i] = targets->data[i] > 0 ? -targets->data[i] / pred * scale : 0;
        }

        // Return gradients for both inputs (prediction gradient and target gradient)
        return {
            std::make_shared<Tensor<T>>(grad_input, predictions->shape, predictions->requires_grad, false),
            std::make_shared<Tensor<T>>(std::vector<T>(targets->data.size(), 0), targets->shape, false, false)  // Target gradient is always zero
        };
    }
};

#endif // CROSS_ENTROPY_LOSS_H 