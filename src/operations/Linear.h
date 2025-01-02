#ifndef LINEAR_H
#define LINEAR_H

#include "Tensor.h"
#include "operations/Operation.h"
#include <memory>
#include <random>

template<typename T>
class Linear : public Operation<T> {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true) 
        : in_features(in_features), out_features(out_features), use_bias(use_bias) {
        // Initialize weights using Xavier/Glorot initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        T bound = std::sqrt(6.0 / (in_features + out_features));
        std::uniform_real_distribution<T> dis(-bound, bound);
        
        std::vector<T> weight_data(in_features * out_features);
        for (auto& w : weight_data) {
            w = dis(gen);
        }
        weights = std::make_shared<Tensor<T>>(
            weight_data,
            std::vector<size_t>{out_features, in_features},
            true,  // requires_grad
            true   // is_leaf
        );

        if (use_bias) {
            std::vector<T> bias_data(out_features, 0);
            bias = std::make_shared<Tensor<T>>(
                bias_data,
                std::vector<size_t>{out_features},
                true,  // requires_grad
                true   // is_leaf
            );
        }
    }

    virtual std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("Linear operation requires exactly 1 input");
        }

        auto input = inputs[0];
        this->saved_tensors = {input};  // Save input for backward pass

        // Check input shape
        if (input->shape.back() != in_features) {
            throw std::invalid_argument("Input features dimension doesn't match");
        }

        // Compute output shape
        std::vector<size_t> output_shape = input->shape;
        output_shape.back() = out_features;

        // Compute matrix multiplication
        std::vector<T> result_data(input->data.size() / in_features * out_features);
        size_t batch_size = input->data.size() / in_features;

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < out_features; ++i) {
                T sum = 0;
                for (size_t j = 0; j < in_features; ++j) {
                    sum += input->data[b * in_features + j] * weights->data[i * in_features + j];
                }
                if (use_bias) {
                    sum += bias->data[i];
                }
                result_data[b * out_features + i] = sum;
            }
        }

        auto result = std::make_shared<Tensor<T>>(
            result_data,
            output_shape,
            input->requires_grad || weights->requires_grad || (bias && bias->requires_grad),
            false  // Not a leaf node
        );

        if (result->requires_grad) {
            result->grad_fn = this->shared_from_this();
        }

        return {result};
    }

    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& grad_outputs) override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("Linear backward pass requires exactly 1 gradient output");
        }

        auto& grad_output = grad_outputs[0];
        auto& input = this->saved_tensors[0];

        // Compute input gradient
        std::vector<T> grad_input_data(input->data.size());
        size_t batch_size = input->data.size() / in_features;

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < in_features; ++j) {
                T sum = 0;
                for (size_t i = 0; i < out_features; ++i) {
                    sum += grad_output->data[b * out_features + i] * weights->data[i * in_features + j];
                }
                grad_input_data[b * in_features + j] = sum;
            }
        }

        // Compute weight gradient
        if (weights->grad == nullptr) {
            weights->grad = std::make_shared<Tensor<T>>(
                std::vector<T>(weights->data.size(), 0),
                weights->shape,
                false,  // requires_grad
                true   // is_leaf
            );
        }

        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                T sum = 0;
                for (size_t b = 0; b < batch_size; ++b) {
                    sum += grad_output->data[b * out_features + i] * input->data[b * in_features + j];
                }
                weights->grad->data[i * in_features + j] += sum;
            }
        }

        // Compute bias gradient
        if (use_bias && bias->grad == nullptr) {
            bias->grad = std::make_shared<Tensor<T>>(
                std::vector<T>(bias->data.size(), 0),
                bias->shape,
                false,  // requires_grad
                true   // is_leaf
            );
        }

        if (use_bias) {
            for (size_t i = 0; i < out_features; ++i) {
                T sum = 0;
                for (size_t b = 0; b < batch_size; ++b) {
                    sum += grad_output->data[b * out_features + i];
                }
                bias->grad->data[i] += sum;
            }
        }

        return {std::make_shared<Tensor<T>>(
            grad_input_data,
            input->shape,
            input->requires_grad,
            false  // Not a leaf node
        )};
    }

    std::shared_ptr<Tensor<T>> weights;
    std::shared_ptr<Tensor<T>> bias;

private:
    size_t in_features;
    size_t out_features;
    bool use_bias;
};

#endif // LINEAR_H 