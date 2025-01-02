#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "Module.h"
#include "operations/Linear.h"

template<typename T>
class LinearLayer : public Module<T> {
public:
    LinearLayer(size_t in_features, size_t out_features, bool use_bias = true) {
        linear_op = std::make_shared<Linear<T>>(in_features, out_features, use_bias);
        
        // Add parameters to the module
        this->params.push_back(linear_op->weights);
        if (use_bias) {
            this->params.push_back(linear_op->bias);
        }
    }

    virtual std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input) override {
        return linear_op->forward({input})[0];
    }

private:
    std::shared_ptr<Linear<T>> linear_op;
};

#endif // LINEAR_LAYER_H 